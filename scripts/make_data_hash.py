#!/usr/bin/env python3
from pathlib import Path
import hashlib


def main() -> None:
    root = Path("data/model_input")
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise SystemExit("No parquet files found under data/model_input")

    lines: list[str] = []
    for p in files:
        h = hashlib.md5()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {p.as_posix()}")

    out = Path("artifacts/hpo/data_model_input.md5")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out} with {len(lines)} entries")


if __name__ == "__main__":
    main()

