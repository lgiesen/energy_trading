"""Fetch day-ahead power prices from the Energy Charts API for multiple bidding zones.

Usage:
    ./.venv/bin/python -m energy_trading.ingestion.fetch_energy_charts \
        --start 2020-11-30T23:00:00Z --end 2025-12-31T23:00:00Z \
        --out data/raw/energy_charts.parquet

Outputs:
    - energy_charts.parquet with hourly UTC timestamps and one price column per zone.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from zoneinfo import ZoneInfo

import polars as pl
import requests

LOGGER = logging.getLogger(__name__)

NEIGHBOR_BZN = [
    "AT",
    "BE",
    "CH",
    "CZ",
    "DK1",
    "DK2",
    "FR",
    "NL",
    # Dropped: da_price_NO2 has ~11.85% nulls, unusable for this project.
    "PL",
    "SE4",
]


def _fetch_zone(base_url: str, zone: str, timeout: int = 60) -> dict:
    url = f"{base_url}&bzn={zone}"
    resp = requests.get(url, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        detail = resp.text[:500] if resp.text else "no response body"
        raise RuntimeError(f"Energy Charts request failed for {zone}: {detail}") from exc
    return resp.json()


def _frame_from_payload(payload: Dict, zone: str) -> pl.DataFrame:
    seconds: List[int] = payload.get("unix_seconds", [])
    prices: List[float] = payload.get("price", [])
    if len(seconds) != len(prices):
        raise ValueError(f"Length mismatch for {zone}: {len(seconds)} timestamps vs {len(prices)} prices")

    ts = pl.from_epoch(pl.Series(seconds, dtype=pl.Int64), time_unit="s").dt.replace_time_zone("UTC")
    df = pl.DataFrame({"timestamp": ts, f"da_price_{zone}": prices})
    # Ensure hourly alignment: average within each hour (handles quarter-hour inputs if ever present)
    df = (
        df.with_columns(pl.col("timestamp").dt.truncate("1h").alias("timestamp"))
        .group_by("timestamp")
        .agg(pl.col(f"da_price_{zone}").mean())
        .sort("timestamp")
    )
    return df


def fetch_all(start: str, end: str, zones: List[str], timeout: int = 60) -> pl.DataFrame:
    base_url = f"https://api.energy-charts.info/price?start={start}&end={end}"
    frames: List[pl.DataFrame] = []
    for zone in zones:
        payload = _fetch_zone(base_url, zone, timeout=timeout)
        frames.append(_frame_from_payload(payload, zone))

    if not frames:
        return pl.DataFrame()

    # Build a unified timestamp index, then join each zone on it to avoid duplicate key columns.
    base = pl.concat([df.select("timestamp") for df in frames]).unique().sort("timestamp")
    wide = base
    for df in frames:
        # coalesce=True avoids duplicate timestamp columns on full join
        wide = wide.join(df, on="timestamp", how="full", coalesce=True)
    return wide.sort("timestamp")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fetch day-ahead prices from Energy Charts API for multiple zones.")
    parser.add_argument("--start", required=True, help="Start ISO8601 (UTC).")
    parser.add_argument("--end", required=True, help="End ISO8601 (UTC).")
    parser.add_argument("--zones", nargs="*", default=NEIGHBOR_BZN, help="Bidding zones to fetch (default: DE-LU + neighbors).")
    parser.add_argument("--out", default="data/raw/energy_charts.parquet", help="Output parquet path.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds per request.")
    args = parser.parse_args()

    start_utc = datetime.fromisoformat(args.start)
    end_utc = datetime.fromisoformat(args.end)
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=timezone.utc)
    else:
        start_utc = start_utc.astimezone(timezone.utc)
    if end_utc.tzinfo is None:
        end_utc = end_utc.replace(tzinfo=timezone.utc)
    else:
        end_utc = end_utc.astimezone(timezone.utc)

    start_local = start_utc.astimezone(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d")
    end_local = end_utc.astimezone(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d")
    df = fetch_all(start_local, end_local, args.zones, timeout=args.timeout)
    if df.is_empty():
        LOGGER.warning("No data fetched.")
        return
    df = (
        df.with_columns(pl.col("timestamp").dt.truncate("1h").alias("timestamp"))
        .filter(
            (pl.col("timestamp") >= pl.lit(start_utc).cast(pl.Datetime(time_unit="us", time_zone="UTC")))
            & (pl.col("timestamp") <= pl.lit(end_utc).cast(pl.Datetime(time_unit="us", time_zone="UTC")))
        )
        .sort("timestamp")
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    LOGGER.info("Wrote %s rows to %s", len(df), out_path)


if __name__ == "__main__":
    main()
