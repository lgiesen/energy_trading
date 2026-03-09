"""Upsample aFRR Regelleistung 15‑minute data to hourly parquet.

Reads data/raw/prices_15min.parquet (from fetch_regelleistung) and writes an
hourly parquet suitable for merge_data. Optionally joins MOL slope features
recomputed from RESULT_LIST_ANONYM_* files.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Re‑use existing MOL processing to avoid duplicate code.
from energy_trading.ingestion.fetch_regelleistung import process_mol_slope

LOGGER = logging.getLogger(__name__)


def _parse_iso(dt_str: str | None) -> datetime | None:
    if not dt_str:
        return None
    dt = datetime.fromisoformat(dt_str)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def upsample_regelleistung(
    prices_15_path: Path,
    out_path: Path,
    start: datetime | None = None,
    end: datetime | None = None,
    mol_dir: Path | None = None,
) -> None:
    if not prices_15_path.exists():
        raise FileNotFoundError(f"15-min prices file not found: {prices_15_path}")

    LOGGER.info("Reading 15-min prices from %s", prices_15_path)
    df = pd.read_parquet(prices_15_path)

    ts_col = "timestamp_utc" if "timestamp_utc" in df.columns else "timestamp"
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()

    LOGGER.info("Resampling to hourly (mean) for numeric columns.")
    df_hour = df.resample("1h").mean()

    if start:
        df_hour = df_hour.loc[df_hour.index >= start]
    if end:
        df_hour = df_hour.loc[df_hour.index <= end]

    # Optional MOL slope (hourly) from anonymous bid lists.
    if mol_dir:
        mol_df = process_mol_slope(Path(mol_dir))
        if not mol_df.empty:
            df_hour = df_hour.join(mol_df, how="left")

    df_hour = df_hour[~df_hour.index.duplicated(keep="first")].sort_index()
    df_hour = df_hour.reset_index().rename(columns={"index": "timestamp_utc"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_hour.to_parquet(out_path, index=False, compression="zstd")
    LOGGER.info("Wrote %s rows to %s", len(df_hour), out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Upsample Regelleistung 15-min prices to hourly parquet.")
    parser.add_argument("--prices-15-in", default="data/raw/prices_15min.parquet", help="Input 15-min parquet path.")
    parser.add_argument("--out", default="data/raw/regelleistung.parquet", help="Hourly parquet output path.")
    parser.add_argument("--start", default=None, help="Optional clip start ISO8601 (UTC).")
    parser.add_argument("--end", default=None, help="Optional clip end ISO8601 (UTC).")
    parser.add_argument("--mol-dir", default="data/raw", help="Directory containing RESULT_LIST_ANONYM_* files.")
    args = parser.parse_args()

    upsample_regelleistung(
        Path(args.prices_15_in),
        Path(args.out),
        start=_parse_iso(args.start),
        end=_parse_iso(args.end),
        mol_dir=Path(args.mol_dir) if args.mol_dir else None,
    )


if __name__ == "__main__":
    main()
