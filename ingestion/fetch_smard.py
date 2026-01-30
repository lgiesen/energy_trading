"""Fetch SMARD load/generation/price series and store them in one parquet file.

Usage:
    python -m ingestion.fetch_smard \
        --start 2020-12-01T00:00:00Z --end 2026-01-01T02:00:00Z \
        --out data/smard.parquet

Outputs:
    - smard.parquet with hourly data aligned on timestamp.

Columns (high level):
    - actuals: load, residual load, wind onshore/offshore, solar
    - forecasts: day-ahead wind/solar, intraday wind onshore
    - prices: da_price_eur (hourly), price_intraday_eur (hourly mean)
    - engineered: forecast errors, wind_forecast_de, total_wind_intraday_error, system_stress_signal
"""
from __future__ import annotations

import argparse
import logging
import warnings
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import polars as pl
import requests

# Suppress urllib3's LibreSSL/OpenSSL compatibility warning on Apple system Python.
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL 1.1.1+.*")

LOGGER = logging.getLogger(__name__)

# SMARD API configuration.
BASE_URL = "https://www.smard.de/app/chart_data"
DEFAULT_REGION = "DE-LU"
DEFAULT_RESOLUTION = "hour"
INTRADAY_REGION = "DE"
INTRADAY_RESOLUTION = "quarterhour"

# Price filter IDs
DA_PRICE_FILTER_ID = 4169  # Day-ahead market price DE/LU
INTRADAY_PRICE_FILTER_ID = 4996  # Intraday trading (quarter-hour)

# SMARD module IDs for non-price series.
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
    # Intraday forecasts
    "wind_onshore_forecast_intraday": 715,
}

# Candidate IDs for realised fossil generation (Ist-Erzeugung).
# We try these IDs in order and keep the first one that returns data.
FOSSIL_GENERATION_CANDIDATES: Dict[str, List[int]] = {
    "generation_fossil_brown_coal_mw": [1223],
    "generation_fossil_hard_coal_mw": [4069],
    "generation_fossil_gas_mw": [4071],
}


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


def _fetch_chunk(
    filter_id: int,
    region: str,
    resolution: str,
    ts: int,
    session: requests.Session,
) -> List[Tuple[int, float]]:
    filename = f"{filter_id}_{region}_{resolution}_{ts}.json"
    url = f"{BASE_URL}/{filter_id}/{region}/{filename}"
    resp = session.get(url, timeout=30)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    return resp.json().get("series", [])


def _series_to_frame(
    name: str,
    series_data: Iterable[Tuple[int, float]],
    start_ms: int,
    end_ms: int,
) -> pl.DataFrame:
    df = pl.DataFrame(series_data, schema=[("timestamp_ms", pl.Int64), (name, pl.Float64)], orient="row")
    df = (
        df.filter((pl.col("timestamp_ms") >= start_ms) & (pl.col("timestamp_ms") <= end_ms))
        .with_columns(pl.from_epoch("timestamp_ms", time_unit="ms").alias("timestamp"))
        .drop("timestamp_ms")
        .group_by("timestamp")
        .agg(pl.col(name).last())
    )
    return df


def fetch_series(
    session: requests.Session,
    filter_id: int,
    region: str,
    resolution: str,
    start_ms: int,
    end_ms: int,
    cutoff_ms: int,
    col_name: str,
) -> pl.DataFrame | None:
    timestamps = _available_timestamps(filter_id, region, resolution, session)
    relevant = [ts for ts in timestamps if ts >= cutoff_ms]
    if not relevant:
        LOGGER.warning("Skipping %s: no timestamps found.", col_name)
        return None

    records: List[Tuple[int, float]] = []
    for ts in relevant:
        try:
            records.extend(_fetch_chunk(filter_id, region, resolution, ts, session))
        except requests.HTTPError as exc:
            LOGGER.warning("Failed chunk %s for %s: %s", ts, col_name, exc)

    if not records:
        LOGGER.warning("Skipping %s: no data records.", col_name)
        return None

    return _series_to_frame(col_name, records, start_ms, end_ms)


def fetch_series_with_candidates(
    session: requests.Session,
    candidates: List[int],
    region: str,
    resolution: str,
    start_ms: int,
    end_ms: int,
    cutoff_ms: int,
    col_name: str,
) -> pl.DataFrame | None:
    """Try multiple filter IDs and return the first non-empty series."""
    for fid in candidates:
        df = fetch_series(session, fid, region, resolution, start_ms, end_ms, cutoff_ms, col_name)
        if df is not None and df.height > 0:
            return df
    LOGGER.warning("Skipping %s: no valid filter_id found in %s", col_name, candidates)
    return None


def fetch_series_with_region_fallback(
    session: requests.Session,
    candidates: List[int],
    regions: List[str],
    resolution: str,
    start_ms: int,
    end_ms: int,
    cutoff_ms: int,
    col_name: str,
) -> pl.DataFrame | None:
    """Fetch series with region fallback (primary first, then fill gaps from fallback)."""
    primary = fetch_series_with_candidates(
        session, candidates, regions[0], resolution, start_ms, end_ms, cutoff_ms, col_name
    )
    fallback = None
    if len(regions) > 1:
        fallback = fetch_series_with_candidates(
            session, candidates, regions[1], resolution, start_ms, end_ms, cutoff_ms, f"{col_name}_fallback"
        )
    if primary is None and fallback is None:
        return None
    if primary is None:
        return fallback.rename({f"{col_name}_fallback": col_name})  # type: ignore[union-attr]
    if fallback is None:
        return primary
    # Coalesce: prefer primary region, fill missing from fallback.
    merged = primary.join(fallback, on="timestamp", how="full", coalesce=True)
    merged = merged.with_columns(
        pl.coalesce([pl.col(col_name), pl.col(f"{col_name}_fallback")]).alias(col_name)
    ).drop(f"{col_name}_fallback")
    return merged


def _resample_qh_to_hour(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
    return (
        df.sort("timestamp")
        .group_by_dynamic("timestamp", every="1h", closed="left", label="left")
        .agg(pl.col(col_name).mean().alias(col_name))
        .sort("timestamp")
    )


def fetch_series_with_resolution_fallback(
    session: requests.Session,
    candidates: List[int],
    region: str,
    start_ms: int,
    end_ms: int,
    cutoff_ms: int,
    col_name: str,
    expected_hours: int,
) -> pl.DataFrame | None:
    """Try hour first; if missing/sparse, fallback to quarterhour and resample."""
    hourly = None
    for fid in candidates:
        hourly = fetch_series(session, fid, region, "hour", start_ms, end_ms, cutoff_ms, col_name)
        if hourly is not None and hourly.height > 0:
            break
    if hourly is not None and hourly.height >= int(0.95 * expected_hours):
        return hourly
    if hourly is not None:
        LOGGER.warning(
            "%s hourly data sparse (%s/%s); trying quarterhour fallback.",
            col_name,
            hourly.height,
            expected_hours,
        )
    # quarterhour fallback
    qh = None
    for fid in candidates:
        qh = fetch_series(session, fid, region, "quarterhour", start_ms, end_ms, cutoff_ms, col_name)
        if qh is not None and qh.height > 0:
            break
    if qh is None:
        return hourly
    return _resample_qh_to_hour(qh, col_name)


def fetch_smard(
    start: datetime,
    end: datetime,
    region: str = DEFAULT_REGION,
    resolution: str = DEFAULT_RESOLUTION,
) -> pl.DataFrame:
    session = _make_session()
    # Convert UTC inputs to Berlin for API-facing calculations (epoch ms stays identical).
    start_api = start.astimezone(ZoneInfo("Europe/Berlin"))
    end_api = end.astimezone(ZoneInfo("Europe/Berlin"))
    start_ms = int(start_api.timestamp() * 1000)
    end_ms = int(end_api.timestamp() * 1000)
    cutoff_ms = int((start - timedelta(days=62)).timestamp() * 1000)

    merged: pl.DataFrame | None = None

    # Non-price series
    for col_name, filter_id in DATA_MODULES.items():
        series_df = fetch_series(session, filter_id, region, resolution, start_ms, end_ms, cutoff_ms, col_name)
        if series_df is None:
            continue
        merged = series_df if merged is None else merged.join(series_df, on="timestamp", how="full")
        if merged is not None and "timestamp_right" in merged.columns:
            merged = merged.drop("timestamp_right")
        LOGGER.info("Fetched %s rows for %s.", len(series_df), col_name)

    if merged is None:
        raise RuntimeError("No SMARD data fetched.")

    # Realised fossil generation (Ist-Erzeugung) with region + resolution fallback
    expected_hours = int((end_ms - start_ms) / 3600000) + 1
    for col_name, candidates in FOSSIL_GENERATION_CANDIDATES.items():
        series_df = None
        # Order: DE-LU @ hour -> DE-LU @ quarterhour -> DE @ hour -> DE @ quarterhour
        for region_try in [DEFAULT_REGION, "DE"]:
            series_df = fetch_series_with_resolution_fallback(
                session,
                candidates,
                region_try,
                start_ms,
                end_ms,
                cutoff_ms,
                col_name,
                expected_hours,
            )
            if series_df is not None and series_df.height > 0:
                if region_try != DEFAULT_REGION:
                    LOGGER.warning("%s fell back to region=%s", col_name, region_try)
                break
        if series_df is None:
            continue
        merged = merged.join(series_df, on="timestamp", how="full", coalesce=True)
        LOGGER.info("Fetched %s rows for %s.", len(series_df), col_name)

    # Day-ahead prices (hourly)
    da_df = fetch_series(session, DA_PRICE_FILTER_ID, region, resolution, start_ms, end_ms, cutoff_ms, "da_price_eur")
    if da_df is not None:
        merged = merged.join(da_df, on="timestamp", how="full", coalesce=True)
        LOGGER.info("Fetched %s rows for da_price_eur.", len(da_df))

    # Intraday prices (quarter-hour) -> hourly mean
    #
    # price_intraday_eur documentation (for thesis):
    # Source: SMARD "Großhandelspreise / Intraday-Handel" (Filter-ID 4996).
    # Methodology: hourly mean of 15-minute volume-weighted average prices (VWAP).
    # Limitation: not an ID1 or ID3 index (closing prices).
    # Assumption: proxy for rebalancing costs in an hourly simulation under liquid markets.
    intraday_qh = fetch_series(
        session,
        INTRADAY_PRICE_FILTER_ID,
        INTRADAY_REGION,
        INTRADAY_RESOLUTION,
        start_ms,
        end_ms,
        cutoff_ms,
        "price_intraday_qh",
    )
    if intraday_qh is not None:
        intraday_hourly = (
            intraday_qh.sort("timestamp")
            .group_by_dynamic("timestamp", every="1h", closed="left", label="left")
            .agg(pl.col("price_intraday_qh").mean().alias("price_intraday_eur"))
            .sort("timestamp")
        )
        merged = merged.join(intraday_hourly, on="timestamp", how="full", coalesce=True)
        LOGGER.info("Fetched %s rows for price_intraday_eur.", len(intraday_hourly))
    # Fill missing intraday prices with day-ahead prices to avoid backtest gaps.
    if "price_intraday_eur" in merged.columns and "da_price_eur" in merged.columns:
        merged = merged.with_columns(
            pl.col("price_intraday_eur").fill_null(pl.col("da_price_eur")).alias("price_intraday_eur")
        )

    merged = merged.sort("timestamp")
    # Enforce strict hourly alignment and clip to requested UTC window.
    merged = merged.with_columns(pl.col("timestamp").dt.truncate("1h").alias("timestamp"))
    start_naive = start.replace(tzinfo=None)
    end_naive = end.replace(tzinfo=None)
    merged = merged.filter(
        (pl.col("timestamp") >= pl.lit(start_naive).cast(pl.Datetime(time_unit="ms")))
        & (pl.col("timestamp") <= pl.lit(end_naive).cast(pl.Datetime(time_unit="ms")))
    )

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
        merged = merged.with_columns((pl.col("solar_forecast") - pl.col("solar_actual")).alias("solar_error"))

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

    # Keep only one UTC and one CET timestamp
    if "timestamp" in merged.columns:
        merged = merged.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC").alias("timestamp_utc"))
        merged = merged.with_columns(pl.col("timestamp").dt.convert_time_zone("Europe/Berlin").alias("timestamp_cet"))
        merged = merged.drop("timestamp")

    # Procured capacity proxy (offered capacity) from capacity overview.
    try:
        try:
            from .fetch_afrr_procured_capacity import fetch_procured_capacity
        except ImportError:
            from energy_trading.ingestion.fetch_afrr_procured_capacity import fetch_procured_capacity
        with requests.Session() as session:
            proc = fetch_procured_capacity(start.isoformat(), end.isoformat(), session)
        if not proc.empty:
            proc_pl = pl.from_pandas(proc.reset_index())
            if "timestamp_utc" not in proc_pl.columns and "timestamp" in proc_pl.columns:
                proc_pl = proc_pl.rename({"timestamp": "timestamp_utc"})
            proc_pl = proc_pl.with_columns(
                pl.col("timestamp_utc")
                .dt.truncate("1h")
                .dt.cast_time_unit("ms")
                .alias("timestamp_utc")
            )
            merged = merged.join(proc_pl, on="timestamp_utc", how="left")
    except Exception as exc:
        LOGGER.warning("Failed to fetch procured capacity proxy: %s", exc)

    return merged


def main() -> None:
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

    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    else:
        start_dt = start_dt.astimezone(timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    else:
        end_dt = end_dt.astimezone(timezone.utc)

    df = fetch_smard(start_dt, end_dt, region=args.region, resolution=args.resolution)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")


if __name__ == "__main__":
    main()
