"""Check data quality for specific columns in entsoe and smard parquet files.

Usage:
    python -m ingestion.check_data_quality
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def _report_header(title: str) -> None:
    print(f"\n--- checking {title} ---")


def _report_col(col: str, series: pd.Series) -> None:
    total = len(series)
    nan_count = int(series.isna().sum())
    is_num = pd.api.types.is_numeric_dtype(series)
    zero_count = int((series == 0).sum()) if is_num else 0
    non_zero = series[(series != 0) & series.notna()] if is_num else series.dropna()

    print(f"[PASS] Column '{col}' found.")
    print(f"       NaNs: {nan_count}")
    if is_num:
        zero_pct = (zero_count / total * 100) if total else 0
        print(f"       Zeros: {zero_count} ({zero_pct:.2f}%)")
        print(f"       Mean: {series.mean():.2f}")
        print(f"       Min: {series.min():.2f}")
        print(f"       Max: {series.max():.2f}")
        if len(non_zero) > 0:
            print(f"       Non-Zero Mean: {non_zero.mean():.2f} MW")
            print("       -> DATA LOOKS GOOD (Contains non-zero values).")
        else:
            print("       -> WARNING: all zeros or NaNs.")
    else:
        print("       Non-numeric column.")


def _check_file(path: Path, columns: list[str], any_match: bool = False) -> None:
    if not path.exists():
        print(f"[FAIL] File not found: {path}")
        return
    df = pd.read_parquet(path)
    _report_header(path.name)
    if any_match:
        found = [c for c in columns if c in df.columns]
        if not found:
            print(f"[FAIL] None of the columns found: {columns}")
            return
        # Report only the first matching column to avoid noisy FAILs.
        _report_col(found[0], df[found[0]])
        return
    for col in columns:
        if col in df.columns:
            _report_col(col, df[col])
        else:
            print(f"[FAIL] Column '{col}' NOT FOUND.")


def _resolve_data_path(filename: str) -> Path:
    candidates = [
        Path("energy_trading/data") / filename,
        Path("data") / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def main() -> None:
    entsoe_path = _resolve_data_path("entsoe.parquet")
    smard_path = _resolve_data_path("smard.parquet")

    entsoe_cols = ["outage_hard_coal_mw", "outage_baseload_mw", "outage_gas_mw"]
    smard_cols = ["price_intraday_eur", "intraday_price_eur", "intraday_price_euro"]

    _check_file(entsoe_path, entsoe_cols)
    _check_file(smard_path, smard_cols, any_match=True)


if __name__ == "__main__":
    main()
