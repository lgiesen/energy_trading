"""Clean and impute energy market datasets with file-specific strategies.

Usage:
    python -m ingestion.clean_datasets --data-dir energy_trading/data --out-dir energy_trading/data/cleaned
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FILES = [
    "commodities.parquet",
    "netztransparenz.parquet",
    "regelleistung.parquet",
    "smard.parquet",
    "entsoe.parquet",
    "energy_charts.parquet",
]


def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "timestamp_utc"})
    if "timestamp_utc" not in df.columns:
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "timestamp_utc"})
        elif "timestamp_cet" in df.columns:
            ts = pd.to_datetime(df["timestamp_cet"])
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize("Europe/Berlin")
            df["timestamp_utc"] = ts.dt.tz_convert("UTC")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def _report(title: str, rows: list[tuple[str, int, int, int]]) -> None:
    print(f"\nFile: {title}")
    print("Column                   | Original NaNs | Filled | Remaining")
    print("-" * 72)
    for col, orig, filled, remain in rows:
        print(f"{col:<24} | {orig:>12} | {filled:>6} | {remain:>9}")


def _ffill_bfill(df: pd.DataFrame, cols: list[str]) -> list[tuple[str, int, int, int]]:
    orig = df[cols].isna().sum()
    df[cols] = df[cols].ffill().bfill()
    after = df[cols].isna().sum()
    rows = []
    for col in cols:
        filled = int(orig[col] - after[col])
        rows.append((col, int(orig[col]), filled, int(after[col])))
    return rows


def _interpolate_time(df: pd.DataFrame, cols: list[str], limit: int) -> list[tuple[str, int, int, int]]:
    orig = df[cols].isna().sum()
    df[cols] = df[cols].interpolate(method="time", limit=limit)
    after = df[cols].isna().sum()
    rows = []
    for col in cols:
        filled = int(orig[col] - after[col])
        rows.append((col, int(orig[col]), filled, int(after[col])))
    return rows


def _numeric_cols(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def clean_file(path: Path, out_dir: Path) -> None:
    df = pd.read_parquet(path)
    df = _ensure_timestamp(df)
    df = df.sort_values("timestamp_utc")
    df = df.set_index("timestamp_utc")

    rows: list[tuple[str, int, int, int]] = []
    name = path.name

    if name == "commodities.parquet":
        cols = _numeric_cols(df, [])
        rows = _ffill_bfill(df, cols)
    elif name == "netztransparenz.parquet":
        cols = _numeric_cols(df, [])
        rows = _interpolate_time(df, cols, limit=6)
    elif name == "entsoe.parquet":
        cols = _numeric_cols(df, [])
        rows = _interpolate_time(df, cols, limit=6)
    elif name == "smard.parquet":
        phys_cols = [c for c in df.columns if c.startswith(("wind_", "solar_", "load_", "residual_load_"))]
        price_cols = [c for c in df.columns if c == "da_price_eur"]
        rows += _interpolate_time(df, phys_cols, limit=6) if phys_cols else []
        rows += _interpolate_time(df, price_cols, limit=2) if price_cols else []
    elif name == "regelleistung.parquet":
        cols = _numeric_cols(df, [])
        rows = _interpolate_time(df, cols, limit=2)
    elif name == "energy_charts.parquet":
        cols = [c for c in df.columns if c.startswith("da_price_")]
        rows = _interpolate_time(df, cols, limit=2) if cols else []
    else:
        return

    _report(name, rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_clean.parquet"
    df.reset_index().to_parquet(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and impute energy datasets.")
    parser.add_argument("--data-dir", default="energy_trading/data", help="Input data directory.")
    parser.add_argument("--out-dir", default="energy_trading/data/cleaned", help="Output directory for cleaned files.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    for name in FILES:
        path = data_dir / name
        if not path.exists():
            print(f"Skipping {name}: not found.")
            continue
        clean_file(path, out_dir)


if __name__ == "__main__":
    main()
