"""Fetch SMARD load/generation/price series and store them in one parquet file.

Usage:
    python -m ingestion.fetch_smard \
        --start 2022-01-01 --end 2025-12-31 \
        --out energy_trading/data/smard.parquet

Outputs:
    - smard.parquet with hourly data aligned on timestamp.

Columns (high level):
    - actuals: load, residual load, wind onshore/offshore, solar
    - forecasts: day-ahead wind/solar, intraday wind onshore
    - prices: day-ahead (hourly)
    - engineered: forecast errors, wind_forecast_de, total_wind_intraday_error, system_stress_signal
"""
from __future__ import annotations

import warnings
import logging

# Suppress urllib3's LibreSSL/OpenSSL compatibility warning on Apple system Python.
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL 1.1.1+.*")

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import polars as pl
import requests
from urllib3.exceptions import NotOpenSSLWarning

# SMARD API configuration.
BASE_URL = "https://www.smard.de/app/chart_data"
DEFAULT_REGION = "DE-LU"
DEFAULT_RESOLUTION = "hour"

# SMARD module IDs.
DATA_MODULES: Dict[str, int] = {
    # Actuals
    "load_actual": 410,
    "residual_load_actual": 4359,
    "wind_onshore_actual": 4067,
    "wind_offshore_actual": 1225,
    "solar_actual": 4068,
    # Forecasts
    "wind_onshore_forecast": 123,
    "wind_offshore_forecast": 3791,
    "solar_forecast": 125,
    # Market price
    "da_price_eur": 4169,  # Day-ahead market price DE/LU
    # Intraday forecasts
    "wind_onshore_forecast_intraday": 715,
}
LOGGER = logging.getLogger(__name__)


def _make_session(retries: int = 3, backoff: float = 0.3) -> requests.Session:
    """Create a requests session with simple retries."""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=requests.adapters.Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff,
            status_forcelist=(500, 502, 503, 504),
        )
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _available_timestamps(filter_id: int, region: str, resolution: str, session: requests.Session) -> List[int]:
    url = f"{BASE_URL}/{filter_id}/{region}/index_{resolution}.json"
    resp = session.get(url, timeout=30)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    payload = resp.json()
    return sorted(payload.get("timestamps", []))


def _fetch_chunk(filter_id: int, region: str, resolution: str, ts: int, session: requests.Session) -> List[Tuple[int, float]]:
    filename = f"{filter_id}_{region}_{resolution}_{ts}.json"
    url = f"{BASE_URL}/{filter_id}/{region}/{filename}"
    resp = session.get(url, timeout=30)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    return resp.json().get("series", [])


def _series_to_frame(name: str, series_data: Iterable[Tuple[int, float]], start_ms: int, end_ms: int) -> pl.DataFrame:
    df = pl.DataFrame(series_data, schema=[("timestamp_ms", pl.Int64), (name, pl.Float64)], orient="row")
    df = (
        df.filter((pl.col("timestamp_ms") >= start_ms) & (pl.col("timestamp_ms") <= end_ms))
        .with_columns(pl.from_epoch("timestamp_ms", time_unit="ms").alias("timestamp"))
        .drop("timestamp_ms")
        .group_by("timestamp")
        .agg(pl.col(name).last())  # keep the last value per timestamp if duplicates exist
    )
    return df


def fetch_smard(
    start: datetime,
    end: datetime,
    region: str = DEFAULT_REGION,
    resolution: str = DEFAULT_RESOLUTION,
) -> pl.DataFrame:
    """Fetch all SMARD series and merge on timestamp."""
    session = _make_session()
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    # Pull a small buffer before start to catch boundary data.
    cutoff_ms = int((start - timedelta(days=62)).timestamp() * 1000)

    merged: pl.DataFrame | None = None
    for col_name, filter_id in DATA_MODULES.items():
        timestamps = _available_timestamps(filter_id, region, resolution, session)
        relevant = [ts for ts in timestamps if ts >= cutoff_ms]
        if not relevant:
            LOGGER.warning("Skipping %s: no timestamps found.", col_name)
            continue

        records: List[Tuple[int, float]] = []
        for ts in relevant:
            try:
                chunk = _fetch_chunk(filter_id, region, resolution, ts, session)
                records.extend(chunk)
            except requests.HTTPError as exc:
                LOGGER.warning("Failed chunk %s for %s: %s", ts, col_name, exc)
        if not records:
            LOGGER.warning("Skipping %s: no data records.", col_name)
            continue

        series_df = _series_to_frame(col_name, records, start_ms, end_ms)
        if merged is None:
            merged = series_df
        else:
            merged = merged.join(series_df, on="timestamp", how="full", suffix=f"_{col_name}")
            # Older Polars versions may produce a duplicate key column named timestamp_right; drop it if present.
            if "timestamp_right" in merged.columns:
                merged = merged.drop("timestamp_right")
        LOGGER.info("Fetched %s rows for %s.", len(series_df), col_name)

    if merged is None:
        raise RuntimeError("No SMARD data fetched.")

    merged = merged.sort("timestamp")
    # Drop duplicate timestamp_* columns created by joins
    ts_cols = [c for c in merged.columns if c.startswith("timestamp_") and c != "timestamp"]
    if ts_cols:
        merged = merged.drop(ts_cols)
    # Derived aggregates
    if "wind_onshore_forecast" in merged.columns and "wind_offshore_forecast" in merged.columns:
        merged = merged.with_columns(
            (pl.col("wind_onshore_forecast") + pl.col("wind_offshore_forecast")).alias("wind_forecast_de")
        )
    # Forecast error features: Forecast - Actual
    if "wind_onshore_forecast" in merged.columns and "wind_onshore_actual" in merged.columns:
        merged = merged.with_columns(
            (pl.col("wind_onshore_forecast") - pl.col("wind_onshore_actual")).alias("wind_onshore_error")
        )
    if "wind_offshore_forecast" in merged.columns and "wind_offshore_actual" in merged.columns:
        merged = merged.with_columns(
            (pl.col("wind_offshore_forecast") - pl.col("wind_offshore_actual")).alias("wind_offshore_error")
        )
    if "solar_forecast" in merged.columns and "solar_actual" in merged.columns:
        merged = merged.with_columns(
            (pl.col("solar_forecast") - pl.col("solar_actual")).alias("solar_error")
        )
    # System stress signal: sum of absolute forecast errors
    if (
        "wind_onshore_error" in merged.columns
        and "wind_offshore_error" in merged.columns
        and "solar_error" in merged.columns
    ):
        merged = merged.with_columns(
            (
                pl.col("wind_onshore_error").abs()
                + pl.col("wind_offshore_error").abs()
                + pl.col("solar_error").abs()
            ).alias("system_stress_signal")
        )

    # Intraday forecast errors (Forecast - Actual)
    if "wind_onshore_forecast_intraday" in merged.columns and "wind_onshore_actual" in merged.columns:
        merged = merged.with_columns(
            (pl.col("wind_onshore_forecast_intraday") - pl.col("wind_onshore_actual")).alias(
                "wind_onshore_intraday_error"
            )
        )
    # Optional fallback: total wind intraday forecast uses offshore day-ahead
    if (
        "wind_onshore_forecast_intraday" in merged.columns
        and "wind_offshore_forecast" in merged.columns
        and "wind_onshore_actual" in merged.columns
        and "wind_offshore_actual" in merged.columns
    ):
        merged = merged.with_columns(
            (pl.col("wind_onshore_forecast_intraday") + pl.col("wind_offshore_forecast")).alias(
                "total_wind_intraday_forecast"
            )
        )
        merged = merged.with_columns(
            (
                pl.col("total_wind_intraday_forecast")
                - (pl.col("wind_onshore_actual") + pl.col("wind_offshore_actual"))
            ).alias("total_wind_intraday_error")
        )

    return merged


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fetch SMARD data and store as parquet.")
    parser.add_argument("--start", default="2022-01-01", help="Start date (UTC, inclusive), e.g. 2022-01-01")
    parser.add_argument("--end", default="2025-12-31", help="End date (UTC, inclusive), e.g. 2025-12-31")
    parser.add_argument("--region", default=DEFAULT_REGION, help="SMARD region code (default DE-LU).")
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION, help="Resolution string used by SMARD (default hour).")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "data" / "smard.parquet"),
        help="Output parquet path.",
    )
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    df = fetch_smard(start_dt, end_dt, region=args.region, resolution=args.resolution)
    # Keep only one UTC and one CET timestamp
    if "timestamp" in df.columns:
        df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC").alias("timestamp_utc"))
        df = df.with_columns(pl.col("timestamp").dt.convert_time_zone("Europe/Berlin").alias("timestamp_cet"))
        df = df.drop("timestamp")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    LOGGER.info("Wrote %s rows to %s", df.height, out_path)


if __name__ == "__main__":
    main()
