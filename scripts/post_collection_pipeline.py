#!/usr/bin/env python3
"""Post-collection pipeline wrapper: refine -> clean -> prune -> transform -> features.

Usage:
    ./.venv/bin/python scripts/post_collection_pipeline.py \
        --input data/processed/all_data.parquet
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
        description="Run post-collection processing (refine -> clean -> prune -> transform -> features)."
    )
    parser.add_argument(
        "--input",
        default="data/processed/all_data.parquet",
        help="Input merged parquet path (default: data/processed/all_data.parquet).",
    )
    parser.add_argument(
        "--refined-out",
        default="data/processed/all_data_refined.parquet",
        help="Refined parquet output.",
    )
    parser.add_argument(
        "--cleaned-out",
        default="data/processed/all_data_cleaned.parquet",
        help="Cleaned parquet output (after missing-value handling).",
    )
    parser.add_argument(
        "--pruned-out",
        default="data/processed/all_data_pruned.parquet",
        help="Pruned parquet output (post redundancy drop).",
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
        "--skip-prune",
        action="store_true",
        help="Stop after clean step.",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip handle_missing_values step and feed refined data directly into prune.",
    )
    parser.add_argument(
        "--skip-transform",
        action="store_true",
        help="Stop after prune step.",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Stop after transform step.",
    )
    args = parser.parse_args()

    py = sys.executable
    Path(args.refined_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.cleaned_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.pruned_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.transformed_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.features_out).parent.mkdir(parents=True, exist_ok=True)

    # 1) Refine merged data (source consolidation + market-specific features).
    _run(
        [
            py,
            "-m",
            "energy_trading.processing.refine_market_data",
            "--in",
            args.input,
            "--out",
            args.refined_out,
        ]
    )

    # 2) Missing-value handling (category-aware imputation + target-row policy).
    prune_input = args.refined_out
    if args.skip_clean:
        LOGGER.info("Skipping clean step (--skip-clean).")
    else:
        _run(
            [
                py,
                "-m",
                "energy_trading.processing.handle_missing_values",
                "--in",
                args.refined_out,
                "--out",
                args.cleaned_out,
            ]
        )
        prune_input = args.cleaned_out

    if args.skip_prune:
        LOGGER.info("Stopping after clean (--skip-prune).")
        return

    # 3) Drop/prune redundant columns.
    _run(
        [
            py,
            "-m",
            "energy_trading.processing.drop_redundant_features",
            "--in",
            prune_input,
            "--out",
            args.pruned_out,
        ]
    )

    if args.skip_transform:
        LOGGER.info("Stopping after prune (--skip-transform).")
        return

    # 4) Transform.
    _run(
        [
            py,
            "-m",
            "energy_trading.processing.transform_data",
            "--in",
            args.pruned_out,
            "--out",
            args.transformed_out,
        ]
    )

    if args.skip_features:
        LOGGER.info("Stopping after transform (--skip-features).")
        return

    # 5) Build features.
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
