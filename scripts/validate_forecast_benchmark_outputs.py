#!/usr/bin/env python3
"""Deprecated compatibility wrapper.
Use scripts/validate_forecast_benchmark.py instead.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--coverage-threshold", type=float, default=0.999)
    _ = ap.parse_args()
    print("[DEPRECATED] Use scripts/validate_forecast_benchmark.py --benchmark-dir <dir>")
    cmd = [
        ".venv/bin/python",
        "scripts/validate_forecast_benchmark.py",
        "--benchmark-dir",
        str(Path(_.output_dir).resolve()),
        "--min-join-coverage",
        str(_.coverage_threshold),
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
