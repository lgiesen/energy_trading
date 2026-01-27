"""Build an hourly master dataset by resampling and merging all source parquets.

Usage:
  python3 -m energy_trading.ingestion.merge_hourly_master \
    --data-dir energy_trading/data \
    --out energy_trading/data/master_energy_data_hourly.parquet
"""
from __future__ import annotations

import argparse
from functools import reduce
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


FILES = [
    "regelleistung.parquet",
    "co2_price.parquet",
    "commodities.parquet",
    "smard.parquet",
    "netztransparenz.parquet",
    "energy_charts.parquet",
    "entsoe.parquet",
]


def _resolve_data_dir(explicit: Path | None) -> Path:
    if explicit and explicit.exists():
        return explicit
    candidates = [
        Path("energy_trading/data"),
        Path("data"),
        Path("../energy_trading/data"),
        Path("../data"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError("Could not locate data directory")


def _ensure_timestamp_utc(df: pd.DataFrame) -> pd.DataFrame:
    ts_col = None
    if "timestamp_utc" in df.columns:
        ts_col = "timestamp_utc"
    else:
        for c in df.columns:
            if "timestamp" in c.lower():
                ts_col = c
                break
    if ts_col is None:
        raise ValueError("No timestamp column found")

    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.drop(columns=[ts_col])
    df.insert(0, "timestamp_utc", ts)
    return df


def _agg_for_column(col: str) -> str:
    cl = col.lower()
    if "price" in cl:
        return "mean"
    if "mwh" in cl or "balance" in cl or "imbalance" in cl:
        return "sum"
    if "mw" in cl:
        return "mean"
    return "mean"


def _prepare_frame(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_parquet(path)
    df = _ensure_timestamp_utc(df)

    # Keep numeric columns only (non-numeric like ticker are dropped for ML features).
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df = df[["timestamp_utc"] + numeric_cols]

    # Drop duplicate timestamps before resampling.
    df = df.drop_duplicates(subset=["timestamp_utc"], keep="last")
    df = df.sort_values("timestamp_utc")
    df = df.set_index("timestamp_utc")

    stem = path.stem
    if stem in {"co2_price", "commodities"}:
        resampled = df.resample("1h").ffill()
    else:
        agg_map = {col: _agg_for_column(col) for col in df.columns}
        resampled = df.resample("1h").agg(agg_map)

    resampled = resampled.reset_index()
    return resampled, [c for c in resampled.columns if c != "timestamp_utc"]


def _merge_frames(frames: Iterable[pd.DataFrame], stems: Iterable[str]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    merged: pd.DataFrame | None = None
    source_cols: Dict[str, List[str]] = {}

    for df, stem in zip(frames, stems):
        if merged is None:
            merged = df.copy()
            source_cols[stem] = [c for c in df.columns if c != "timestamp_utc"]
            continue

        rename: Dict[str, str] = {}
        existing = set(merged.columns)
        for c in df.columns:
            if c == "timestamp_utc":
                continue
            if c in existing:
                rename[c] = f"{c}_{stem}"
        if rename:
            df = df.rename(columns=rename)
        source_cols[stem] = [rename.get(c, c) for c in df.columns if c != "timestamp_utc"]
        merged = pd.merge(merged, df, on="timestamp_utc", how="outer")

    if merged is None:
        raise RuntimeError("No frames to merge")

    return merged, source_cols


def build_hourly_master(data_dir: Path, out_path: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    stems: List[str] = []
    for fname in FILES:
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        frame, _ = _prepare_frame(path)
        frames.append(frame)
        stems.append(path.stem)

    merged, source_cols = _merge_frames(frames, stems)

    # Forward-fill CO2/commodities after merge to bridge gaps.
    for stem in ("co2_price", "commodities"):
        cols = source_cols.get(stem, [])
        if cols:
            merged[cols] = merged[cols].ffill()

    merged = merged.sort_values("timestamp_utc")

    # Add CET timestamps for analysis convenience.
    merged["timestamp_cet"] = merged["timestamp_utc"].dt.tz_convert("Europe/Berlin")
    merged["timestamp_cet_naive"] = merged["timestamp_cet"].dt.tz_localize(None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False, compression="zstd")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hourly master dataset from all parquet sources.")
    parser.add_argument("--data-dir", default=None, help="Directory containing parquet files.")
    parser.add_argument(
        "--out",
        default="energy_trading/data/master_energy_data_hourly.parquet",
        help="Output parquet path.",
    )
    args = parser.parse_args()

    data_dir = _resolve_data_dir(Path(args.data_dir) if args.data_dir else None)
    out_path = Path(args.out)
    merged = build_hourly_master(data_dir, out_path)
    print(f"Wrote {len(merged)} rows to {out_path}")


if __name__ == "__main__":
    main()
