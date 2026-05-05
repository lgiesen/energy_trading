#!/usr/bin/env python3
"""Package a model run directory into a deliverable zip."""

from __future__ import annotations

import sys
from pathlib import Path
import zipfile


def main() -> None:
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: python3 scripts/package_audit.py <run_id>")

    run_id = sys.argv[1].strip()
    if not run_id:
        raise RuntimeError("run_id must not be empty")

    run_dir = Path("artifacts/model_runs") / run_id
    if not run_dir.exists():
        raise RuntimeError(f"Run directory not found: {run_dir}")

    zip_path = Path("artifacts/model_runs") / f"{run_id}_deliverable.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(run_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(run_dir.parent))

    print(f"[OK] wrote {zip_path}")


if __name__ == "__main__":
    main()

