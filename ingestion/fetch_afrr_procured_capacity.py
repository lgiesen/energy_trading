"""Fetch aFRR procured capacity (SRL Bezuschlagte Leistung) and merge into smard.parquet.

Usage:
    python -m energy_trading.ingestion.fetch_afrr_procured_capacity \\
        --start 2020-12-01T00:00:00Z --end 2026-01-01T02:00:00Z \\
        --smard data/raw/smard.parquet
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests


LOGGER = logging.getLogger(__name__)
BASE_URL = "https://www.regelleistung.net/apps/cpp-publisher/api/v2/tenders/files"

# Suppress noisy openpyxl default style warnings from Regelleistung files.
warnings.filterwarnings(
    "ignore",
    message="Workbook contains no default style, apply openpyxl's default",
    category=UserWarning,
    module="openpyxl",
)


def _pick_first(d: dict[str, Any], keys: Iterable[str]) -> Any | None:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _parse_direction(val: Any) -> str | None:
    if val is None:
        return None
    s = str(val).strip().upper()
    if s in {"POS", "POSITIVE", "UP", "POSITIVE_DIRECTION"}:
        return "pos"
    if s in {"NEG", "NEGATIVE", "DOWN", "NEGATIVE_DIRECTION"}:
        return "neg"
    if "POS" in s and "NEG" not in s:
        return "pos"
    if "NEG" in s:
        return "neg"
    return None


def _to_utc(ts: pd.Series, source_tz: str) -> pd.Series:
    dt = pd.to_datetime(ts, errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(source_tz, ambiguous="infer", nonexistent="shift_forward")
    return dt.dt.tz_convert("UTC")


def _find_col(df: pd.DataFrame, needles: list[str]) -> str | None:
    for c in df.columns:
        c_up = str(c).upper()
        if all(n.upper() in c_up for n in needles):
            return c
    return None


def _parse_capacity_year(year: int, session: requests.Session) -> pd.DataFrame:
    """Download capacity market overview and return a tidy hourly frame."""
    url = f"{BASE_URL}/RESULT_OVERVIEW_CAPACITY_MARKET_aFRR_{year}-01-01_{year}-12-31.xlsx"
    resp = session.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    if resp.status_code != 200:
        LOGGER.info("Capacity overview file missing for %s (status %s).", year, resp.status_code)
        return pd.DataFrame()

    df = pd.read_excel(pd.io.common.BytesIO(resp.content), engine="openpyxl")
    df.columns = df.columns.str.strip()

    date_col = _find_col(df, ["DATE", "FROM"]) or _find_col(df, ["DELIVERY", "DATE"]) or df.columns[0]
    prod_col = _find_col(df, ["PRODUCT"]) or df.columns[3]
    cap_col = _find_col(df, ["GERMANY", "OFFERED", "CAPACITY"]) or _find_col(df, ["OFFERED", "CAPACITY"])
    if cap_col is None:
        LOGGER.warning("No capacity column found for %s.", year)
        return pd.DataFrame()

    df = df[[date_col, prod_col, cap_col]].copy()
    df[cap_col] = pd.to_numeric(df[cap_col], errors="coerce")
    df = df.dropna(subset=[date_col, prod_col, cap_col])

    prod_str = df[prod_col].astype(str)
    mask_block = prod_str.str.contains(r"_\\d{2}_\\d{2}")
    mask_qh = prod_str.str.contains(r"_\\d{3}") & ~mask_block

    def _process_part(df_part: pd.DataFrame, mode: str) -> pd.DataFrame:
        df_part = df_part.copy()
        start_of_day = pd.to_datetime(df_part[date_col], errors="coerce").dt.tz_localize(
            "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward"
        )
        if mode == "block":
            df_part["start_hour"] = df_part[prod_col].str.extract(r"_(\\d{2})_").astype(int)
            df_part["timestamp"] = start_of_day + pd.to_timedelta(df_part["start_hour"], unit="h")
            df_part["direction"] = df_part[prod_col].str.split("_").str[0]
            df_part = df_part.dropna(subset=["timestamp"])
        elif mode == "qh":
            df_part["quarter_hour_int"] = df_part[prod_col].str.extract(r"_(\\d+)").astype(int)
            df_part["timestamp"] = start_of_day + pd.to_timedelta((df_part["quarter_hour_int"] - 1) * 15, unit="m")
            df_part["direction"] = df_part[prod_col].str.split("_").str[0]
            df_part = df_part.dropna(subset=["timestamp"])
        else:
            return pd.DataFrame()

        df_part["timestamp"] = df_part["timestamp"].dt.tz_convert("UTC")
        df_part = df_part[["timestamp", "direction", cap_col]]
        df_pivot = df_part.pivot_table(index="timestamp", columns="direction", values=cap_col, aggfunc="last")
        if mode == "block":
            df_pivot = df_pivot.resample("1h").ffill(limit=3)
        else:
            df_pivot = df_pivot.resample("1h").mean()
        return df_pivot

    parts = []
    if mask_block.any():
        parts.append(_process_part(df[mask_block], "block"))
    if mask_qh.any():
        parts.append(_process_part(df[mask_qh], "qh"))
    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def fetch_procured_capacity(start: str, end: str, session: requests.Session) -> pd.DataFrame:
    """Fetch procured capacity proxy from regelleistung capacity results."""
    start_year = pd.to_datetime(start).year
    end_year = pd.to_datetime(end).year
    frames = []
    for y in range(start_year, end_year + 1):
        frames.append(_parse_capacity_year(y, session))

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames).sort_index()
    df = df.loc[pd.to_datetime(start, utc=True): pd.to_datetime(end, utc=True)]
    df = df.rename(columns={"POS": "afrr_procured_capacity_mw_pos", "NEG": "afrr_procured_capacity_mw_neg"})
    # Some files may use lowercase or different direction labels.
    if "pos" in df.columns:
        df = df.rename(columns={"pos": "afrr_procured_capacity_mw_pos"})
    if "neg" in df.columns:
        df = df.rename(columns={"neg": "afrr_procured_capacity_mw_neg"})
    return df


def expand_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Expand interval rows to hourly rows."""
    rows = []
    for _, r in df.iterrows():
        start = r["start_utc"]
        end = r["end_utc"]
        if start >= end:
            continue
        hours = pd.date_range(start=start, end=end, freq="1h", inclusive="left", tz="UTC")
        if len(hours) == 0:
            continue
        for ts in hours:
            row = r.copy()
            row["timestamp_utc"] = ts
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def merge_procured_into_smard(smard_path: Path, procured: pd.DataFrame) -> None:
    df = pd.read_parquet(smard_path)
    if "timestamp_utc" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "timestamp_utc"})
    merged = df.merge(procured, how="left", on="timestamp_utc")
    merged.to_parquet(smard_path, index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fetch aFRR procured capacity and merge into smard.parquet.")
    parser.add_argument("--start", required=True, help="Start ISO8601 (UTC).")
    parser.add_argument("--end", required=True, help="End ISO8601 (UTC).")
    parser.add_argument("--smard", required=True, help="Path to smard.parquet to update.")
    args = parser.parse_args()

    LOGGER.info("Fetching aFRR procured capacity proxy (offered capacity) from %s to %s", args.start, args.end)
    with requests.Session() as session:
        procured = fetch_procured_capacity(args.start, args.end, session)
    if procured.empty:
        LOGGER.warning("No procured capacity data fetched.")
        return
    procured = procured.reset_index().rename(columns={"index": "timestamp_utc"})
    merge_procured_into_smard(Path(args.smard), procured)
    LOGGER.info("Wrote updated smard.parquet with procured capacity columns: %s", args.smard)


if __name__ == "__main__":
    main()
