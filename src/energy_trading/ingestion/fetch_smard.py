"""Fetch SMARD load/generation/price series and store them in one parquet file.

Usage:
    ./.venv/bin/python -m energy_trading.ingestion.fetch_smard \
        --start 2020-11-30T23:00:00Z --end 2025-12-31T23:00:00Z \
        --out data/raw/smard.parquet
        --market-data-out data/raw/installed_capacity.csv

    Disable CSV download (timeseries-only):
    ./.venv/bin/python -m energy_trading.ingestion.fetch_smard \
        --start 2020-11-30T23:00:00Z --end 2025-12-31T23:00:00Z \
        --out data/raw/smard.parquet \
        --skip-market-data-csv

Outputs:
    - data/raw/smard.parquet (hourly SMARD timeseries + joined installed capacity columns)
    - data/raw/installed_capacity.csv (downloaded binary CSV from nip-download-manager API)

Columns (high level):
    - actuals: residual load, wind onshore/offshore, solar
    - forecasts: day-ahead wind/solar
    - generation by fuel: lignite, hard coal, gas, nuclear, hydro pumped storage
    - prices: da_price (hourly)
    - engineered: forecast errors, wind_forecast_de, system_stress_signal

API notes:
    - chart_data API (GET): continuous timeseries used for smard.parquet
      Endpoint pattern: https://www.smard.de/app/chart_data/{filter_id}/{region}/{file}.json
    - nip-download-manager API (POST): bulk market-data export used for installed_capacity.csv
      Endpoint: https://www.smard.de/nip-download-manager/nip/download/market-data
    - Default behavior in this script: run both APIs in one command.
"""
from __future__ import annotations

import argparse
import logging
import warnings
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import polars as pl
import requests

# Suppress urllib3's LibreSSL/OpenSSL compatibility warning on Apple system Python.
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL 1.1.1+.*")

LOGGER = logging.getLogger(__name__)

# SMARD API configuration.
BASE_URL = "https://www.smard.de/app/chart_data"
DEFAULT_REGION = "DE-LU"
DEFAULT_RESOLUTION = "hour"
# Price filter IDs
DA_PRICE_FILTER_ID = 4169  # Day-ahead market price DE/LU

# SMARD module IDs for non-price series.
DATA_MODULES: Dict[str, int] = {
    # Actuals (load_actual dropped; sourced from ENTSO-E)
    "residual_load_actual": 4359,
    "wind_offshore_actual": 1225,
    "solar_actual": 4068,
    # Forecasts
    "wind_onshore_forecast": 123,
    "wind_offshore_forecast": 3791,
    "solar_forecast": 125,
}

MARKET_CONFIG_URL = "https://www.smard.de/app/chart_configuration/market_data_configuration.json"

# Candidate IDs for realised generation (Ist-Erzeugung).
# We try these IDs in order and keep the first one that returns data.
GENERATION_CANDIDATES: Dict[str, List[int]] = {
    "generation_fossil_brown_coal_mw": [1223],
    "generation_fossil_hard_coal_mw": [4069],
    "generation_fossil_gas_mw": [4071],
    # Quarterhour IDs to avoid gaps.
    "generation_nuclear_mw": [1224],
    # 4070 = Stromerzeugung: Pumpspeicher (4066 is Biomasse)
    "generation_hydro_pumped_storage_mw": [4070],
}

MARKET_DATA_INIT_URL = "https://www.smard.de/en/downloadcenter/download-market-data/"
MARKET_DATA_POST_URL = "https://www.smard.de/nip-download-manager/nip/download/market-data"
MARKET_DATA_PAYLOAD = {
    "request_form": [
        {
            "format": "CSV",
            "moduleIds": [
                3004073,
                3004076,
                3004072,
                3004074,
                3004075,
                3000186,
                3000188,
                3000189,
                3000194,
                3000198,
                3003792,
                3000207,
            ],
            "region": "DE-LU",
            "timestamp_from": 1609455600000,
            "timestamp_to": 1769900400000,
            "type": "discrete",
            "language": "en",
            "resolution": "hour",
        }
    ]
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


def download_market_data_csv(out_path: Path) -> None:
    """Replicate SMARD market-data POST download and save CSV as binary stream."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/144.0.0.0 Safari/537.36"
        )
        # Initialize session/cookies.
        session.get(MARKET_DATA_INIT_URL, headers={"User-Agent": user_agent}, timeout=60).raise_for_status()

        headers = {
            "Content-Type": "application/json",
            "User-Agent": user_agent,
            "Referer": MARKET_DATA_INIT_URL,
        }
        resp = session.post(
            MARKET_DATA_POST_URL,
            headers=headers,
            json=MARKET_DATA_PAYLOAD,
            timeout=180,
            stream=True,
        )
        status_code = resp.status_code
        resp.raise_for_status()

        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"Downloaded {out_path} (status={status_code}, bytes={out_path.stat().st_size})")


def load_installed_capacity_csv(
    csv_path: Path,
    start: datetime,
    end: datetime,
) -> pl.DataFrame | None:
    """Load SMARD market-data CSV and return selected capacity columns on timestamp_utc."""
    if not csv_path.exists():
        LOGGER.info("Installed capacity CSV not found at %s (skip join).", csv_path)
        return None

    # SMARD export may prepend metadata lines and place real header later.
    raw_text = csv_path.read_text(encoding="utf-8-sig", errors="replace")
    lines = raw_text.splitlines()
    header_idx = None
    delimiter = ";"
    for i, line in enumerate(lines):
        l = line.strip()
        if not l:
            continue
        if l.startswith("Start date;") or l.startswith("Start date\t"):
            header_idx = i
            delimiter = "\t" if "\t" in l and ";" not in l else ";"
            break
    if header_idx is None:
        LOGGER.warning("Installed capacity CSV header row not found in %s", csv_path)
        return None

    df = pd.read_csv(
        csv_path,
        sep=delimiter,
        encoding="utf-8-sig",
        skiprows=header_idx,
        header=0,
    )
    df.columns = [str(c).strip() for c in df.columns]
    if "Start date" not in df.columns:
        LOGGER.warning("Installed capacity CSV missing 'Start date' column after parsing: %s", csv_path)
        return None

    mapping_candidates = {
        "wind_onshore_capacity": [
            "Wind onshore [MW]",
            "Wind onshore [MW] Calculated resolutions",
        ],
        "wind_offshore_capacity": [
            "Wind offshore [MW]",
            "Wind offshore [MW] Calculated resolutions",
        ],
        "solar_capacity": [
            "Photovoltaics [MW]",
            "Photovoltaics [MW] Calculated resolutions",
        ],
        "gas_capacity": [
            "Fossil gas [MW]",
            "Fossil gas [MW] Calculated resolutions",
        ],
        "hard_coal_capacity": [
            "Hard coal [MW]",
            "Hard coal [MW] Calculated resolutions",
        ],
        "lignite_capacity": [
            "Lignite [MW]",
            "Lignite [MW] Calculated resolutions",
        ],
        "pumped_storage_capacity": [
            "Hydro pumped storage [MW]",
            "Hydro pumped storage [MW] Calculated resolutions",
        ],
    }
    available: Dict[str, str] = {}
    for out_col, candidates in mapping_candidates.items():
        src = next((c for c in candidates if c in df.columns), None)
        if src:
            available[src] = out_col
    if not available:
        LOGGER.warning("Installed capacity CSV has none of the expected capacity columns: %s", csv_path)
        return None

    keep_cols = ["Start date"] + list(available.keys())
    df = df[keep_cols].rename(columns={"Start date": "timestamp_utc", **available})
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        format="%b %d, %Y %I:%M %p",
        errors="coerce",
    )
    # CSV timestamps are in local German market time.
    df["timestamp_utc"] = (
        df["timestamp_utc"]
        .dt.tz_localize("Europe/Berlin", ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )

    cap_cols = list(available.values())
    for col in cap_cols:
        s = df[col].astype(str).str.strip()
        s = s.replace("-", pd.NA)
        # Example values: "8,428.00" -> 8428.00
        s = s.str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(s, errors="coerce")

    # Keep only requested window and one value per hour.
    df = df[(df["timestamp_utc"] >= start) & (df["timestamp_utc"] <= end)]
    df["timestamp_utc"] = df["timestamp_utc"].dt.floor("h")
    df = df.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"], keep="last")
    if df.empty:
        return None

    return pl.from_pandas(df)


def _available_timestamps(filter_id: int, region: str, resolution: str, session: requests.Session) -> List[int]:
    url = f"{BASE_URL}/{filter_id}/{region}/index_{resolution}.json"
    resp = session.get(url, timeout=30)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    payload = resp.json()
    return sorted(payload.get("timestamps", []))


def _lookup_data_id_by_name(
    session: requests.Session,
    name_contains: str,
    resolution: str = "quarterhour",
) -> List[int]:
    """Lookup SMARD data_id by module name from market_data_configuration.json."""
    try:
        resp = session.get(MARKET_CONFIG_URL, timeout=30)
        resp.raise_for_status()
        cfg = resp.json()
    except Exception:
        return []

    matches: List[int] = []

    def walk(obj) -> None:
        if isinstance(obj, dict):
            if "data_id" in obj and "name" in obj:
                name = str(obj.get("name", "")).lower()
                if name_contains.lower() in name:
                    if resolution is None or obj.get("source_resolution") == resolution:
                        matches.append(int(obj["data_id"]))
            for val in obj.values():
                walk(val)
        elif isinstance(obj, list):
            for val in obj:
                walk(val)

    walk(cfg)
    # Deduplicate while preserving order.
    seen = set()
    out = []
    for mid in matches:
        if mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out


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


def _resample_qh_to_hour(df: pl.DataFrame, col_name: str, agg: str = "sum") -> pl.DataFrame:
    """Resample quarter-hour series to hourly.

    Aggregation policy:
    - energy/volume series in [MWh]: use `sum`
    - price/index series: use `mean`
    """
    if agg not in {"sum", "mean"}:
        raise ValueError(f"Unsupported qh->hour aggregation: {agg}")
    agg_expr = pl.col(col_name).sum() if agg == "sum" else pl.col(col_name).mean()
    return (
        df.sort("timestamp")
        .group_by_dynamic("timestamp", every="1h", closed="left", label="left")
        .agg(agg_expr.alias(col_name))
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
    qh_to_hour_agg: str = "sum",
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
    return _resample_qh_to_hour(qh, col_name, agg=qh_to_hour_agg)


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

    # Wind onshore actuals (region fallback: DE first, then DE-LU)
    wind_onshore_actual = fetch_series_with_region_fallback(
        session,
        [4067],
        regions=["DE", "DE-LU"],
        resolution="hour",
        start_ms=start_ms,
        end_ms=end_ms,
        cutoff_ms=cutoff_ms,
        col_name="wind_onshore_actual",
    )
    if wind_onshore_actual is not None:
        merged = merged.join(wind_onshore_actual, on="timestamp", how="full", coalesce=True)
        LOGGER.info("Fetched %s rows for wind_onshore_actual.", len(wind_onshore_actual))

    # Realised generation (Ist-Erzeugung) with region + resolution fallback
    expected_hours = int((end_ms - start_ms) / 3600000) + 1
    for col_name, candidates in GENERATION_CANDIDATES.items():
        series_df = None
        # Nuclear + pumped storage are delivered as energy series; aggregate qh->hour by sum.
        if col_name in {"generation_nuclear_mw", "generation_hydro_pumped_storage_mw"}:
            for region_try in [DEFAULT_REGION, "DE"]:
                series_df = fetch_series_with_candidates(
                    session,
                    candidates,
                    region_try,
                    "quarterhour",
                    start_ms,
                    end_ms,
                    cutoff_ms,
                    col_name,
                )
                if series_df is not None and series_df.height > 0:
                    series_df = _resample_qh_to_hour(series_df, col_name, agg="sum")
                    if region_try != DEFAULT_REGION:
                        LOGGER.warning("%s fell back to region=%s", col_name, region_try)
                    break
        else:
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
                    qh_to_hour_agg="sum",
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
    da_df = fetch_series(session, DA_PRICE_FILTER_ID, region, resolution, start_ms, end_ms, cutoff_ms, "da_price")
    if da_df is not None:
        merged = merged.join(da_df, on="timestamp", how="full", coalesce=True)
        LOGGER.info("Fetched %s rows for da_price.", len(da_df))

    # TODO: Placeholder for profit analysis. Currently a copy of DA price.
    # Replace with actual EPEX ID1 data later.
    if "da_price" in merged.columns:
        merged = merged.with_columns(pl.col("da_price").alias("price_intraday_eur"))

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

    # Keep only one UTC and one CET timestamp
    if "timestamp" in merged.columns:
        merged = merged.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC").alias("timestamp_utc"))
        merged = merged.with_columns(pl.col("timestamp").dt.convert_time_zone("Europe/Berlin").alias("timestamp_cet"))
        merged = merged.drop("timestamp")

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
        default=str(Path(__file__).resolve().parents[3] / "data" / "raw" / "smard.parquet"),
        help="Output parquet path.",
    )
    parser.add_argument(
        "--skip-market-data-csv",
        action="store_true",
        help="Skip market-data CSV download and only use chart_data API.",
    )
    parser.add_argument(
        "--market-data-out",
        default=str(Path(__file__).resolve().parents[3] / "data" / "raw" / "installed_capacity.csv"),
        help="Output path for SMARD market-data CSV download.",
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

    if not args.skip_market_data_csv:
        download_market_data_csv(Path(args.market_data_out))

    df = fetch_smard(start_dt, end_dt, region=args.region, resolution=args.resolution)
    cap_df = load_installed_capacity_csv(Path(args.market_data_out), start_dt, end_dt)
    if cap_df is not None and cap_df.height > 0:
        # Align join-key dtype with SMARD frame (datetime[ms, UTC]).
        cap_df = cap_df.with_columns(
            pl.col("timestamp_utc").dt.cast_time_unit("ms").alias("timestamp_utc")
        )
        df = df.join(cap_df, on="timestamp_utc", how="left", coalesce=True)
        LOGGER.info("Joined installed capacity CSV (%s rows) into smard parquet.", cap_df.height)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Drop SMARD load duplicate if it exists (we use ENTSO-E load_actual).
    if "load_actual_smard" in df.columns:
        df = df.drop("load_actual_smard")
    df.write_parquet(out_path, compression="zstd")

if __name__ == "__main__":
    main()
