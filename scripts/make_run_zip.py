#!/usr/bin/env python3
from pathlib import Path
import argparse
import zipfile


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create run deliverable zip.")
    p.add_argument("--run-id", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    run_dir = Path("artifacts/model_runs") / run_id
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    zip_path = Path("artifacts/model_runs") / f"{run_id}_deliverable.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(run_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(run_dir.parent))
    print(f"[OK] wrote {zip_path}")


if __name__ == "__main__":
    main()

