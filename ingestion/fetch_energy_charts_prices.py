"""Fetch day-ahead power prices from the Energy Charts API for multiple bidding zones.

Usage:
    python -m ingestion.fetch_energy_charts_prices \
        --start 2022-01-01 --end 2025-12-31 \
        --out data/energy_charts.parquet

Outputs:
    - energy_charts.parquet with hourly UTC timestamps and one price column per zone.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import polars as pl
import requests


NEIGHBOR_BZN = [
    "AT",
    "BE",
    "CH",
    "CZ",
    "DK1",
    "DK2",
    "FR",
    "NL",
    "NO2",
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
    parser = argparse.ArgumentParser(description="Fetch day-ahead prices from Energy Charts API for multiple zones.")
    parser.add_argument("--start", required=True, help="Start ISO8601, e.g. 2022-01-01T00:00Z")
    parser.add_argument("--end", required=True, help="End ISO8601, e.g. 2022-02-01T00:00Z")
    parser.add_argument("--zones", nargs="*", default=NEIGHBOR_BZN, help="Bidding zones to fetch (default: DE-LU + neighbors).")
    parser.add_argument("--out", default="data/energy_charts.parquet", help="Output parquet path.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds per request.")
    args = parser.parse_args()

    df = fetch_all(args.start, args.end, args.zones, timeout=args.timeout)
    if df.is_empty():
        print("No data fetched.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
