"""Inspect raw Regelleistung ENERGY overview files for time segmentation issues.

Usage:
    python -m ingestion.inspect_raw_regelleistung --file RESULT_OVERVIEW_ENERGY_MARKET_aFRR_2022-01-01_2022-01-31.xlsx

If --file is omitted, the script searches the data directory for matching files.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import requests


def _find_col(df: pd.DataFrame, needles: list[str]) -> str | None:
    for c in df.columns:
        c_up = str(c).upper()
        if all(n.upper() in c_up for n in needles):
            return c
    return None


def _load_excel(path_or_url: str) -> pd.DataFrame:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        resp = requests.get(path_or_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return pd.read_excel(io.BytesIO(resp.content), engine="openpyxl")
    return pd.read_excel(path_or_url, engine="openpyxl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw Regelleistung ENERGY overview file structure.")
    parser.add_argument(
        "--file",
        default="RESULT_OVERVIEW_ENERGY_MARKET_aFRR_2022-01-01_2022-01-31.xlsx",
        help="Filename or URL to inspect.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Directory to search for the file if not found by name.",
    )
    args = parser.parse_args()

    file_arg = args.file
    path = Path(file_arg)
    if not path.exists() and not file_arg.startswith(("http://", "https://")):
        data_dir = Path(args.data_dir)
        matches = list(data_dir.rglob(file_arg))
        if not matches:
            print(f"File not found: {file_arg}")
            print(f"Searched in: {data_dir}")
            print("Hint: pass a full path or URL with --file.")
            return
        path = matches[0]
        if len(matches) > 1:
            print(f"Multiple matches found. Using: {path}")

    df = _load_excel(str(path))
    df.columns = df.columns.str.strip()

    print("Columns:")
    print(df.columns.tolist())

    date_col = _find_col(df, ["DATE"]) or _find_col(df, ["DATUM"])
    time_col = _find_col(df, ["TIME"]) or _find_col(df, ["ZEIT"]) or _find_col(df, ["PERIOD"])
    product_col = _find_col(df, ["PRODUCT"]) or _find_col(df, ["PRODUKT"]) or _find_col(df, ["ZEITSCHEIBE"])

    print("\nDetected columns:")
    print(f"  date_col: {date_col}")
    print(f"  time_col: {time_col}")
    print(f"  product_col: {product_col}")

    print("\nHead (first 20 rows):")
    print(df.head(20))

    if date_col:
        total_rows = len(df)
        unique_dates = df[date_col].nunique(dropna=True)
        print(f"\nRow count: {total_rows}")
        print(f"Unique {date_col}: {unique_dates}")
        if unique_dates:
            ratio = total_rows / unique_dates
            print(f"Rows per date: {ratio:.2f}")
        if product_col:
            print("\nSample product values:")
            print(df[product_col].dropna().astype(str).unique()[:10])


if __name__ == "__main__":
    main()
