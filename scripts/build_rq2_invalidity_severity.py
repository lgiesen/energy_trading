#!/usr/bin/env python3
"""Compatibility wrapper for the canonical simulation invalidity extractor."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from scripts.build_simulation_invalidity_severity import (
    DEFAULT_RUN_ROOT,
    build_outputs as build_generic_outputs,
)


DEFAULT_OUT_DIR = Path("artifacts/benchmark/rq2_simulation_benchmark")


def _write_legacy_aliases(paths: dict[str, Path], out_dir: Path) -> dict[str, Path]:
    aliases = {
        "summary": out_dir / "backup" / "diagnostics" / "rq2_invalidity_severity_summary.csv",
        "hourly": out_dir / "backup" / "diagnostics" / "rq2_invalidity_severity_by_hour.csv",
        "inventory": out_dir / "backup" / "diagnostics" / "rq2_invalidity_source_inventory.csv",
        "warnings": out_dir / "backup" / "warnings" / "rq2_invalidity_severity_warnings.csv",
    }
    for key, dst in aliases.items():
        src = paths[key]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return {**paths, **{f"rq2_{k}": v for k, v in aliases.items()}}


def build_outputs(run_dir: Path, out_dir: Path) -> dict[str, Path]:
    paths = build_generic_outputs(run_dir, out_dir, label="rq2")
    return _write_legacy_aliases(paths, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deprecated RQ2 wrapper. Use scripts/build_simulation_invalidity_severity.py for new work."
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = build_outputs(Path(args.run_dir), Path(args.out_dir))
    for name, path in paths.items():
        print(f"[OK] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
