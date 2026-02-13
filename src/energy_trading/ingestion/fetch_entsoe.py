"""Fetch ENTSO-E wind/solar actuals and forecasts for DE-LU using entsoe-py.

Usage:
    ./.venv/bin/python -m energy_trading.ingestion.fetch_entsoe \
        --start 2020-12-01T00:00:00Z --end 2026-01-01T02:00:00Z \
        --out data/raw/entsoe.parquet

Outputs:
    - entsoe.parquet with hourly UTC timestamps and:
        timestamp_utc,
        wind_onshore_actual_entsoe, wind_offshore_actual_entsoe, solar_actual_entsoe,
        wind_onshore_forecast_da_entsoe, wind_offshore_forecast_da_entsoe, solar_forecast_da_entsoe,
        wind_onshore_forecast_id_entsoe, wind_offshore_forecast_id_entsoe, solar_forecast_id_entsoe
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl
from entsoe import EntsoePandasClient

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

LOGGER = logging.getLogger(__name__)

# Domain codes (kept explicit to avoid mixing bidding-zone and physical/control areas):
# - DE_LU_BIDDING_ZONE_CODE (10Y1001A1001A82H): use for price endpoints
# - DE_PHYSICAL_CONTROL_CODE (10Y1001A1001A83F): use for generation/capacity/load
DE_LU_BIDDING_ZONE_CODE = "10Y1001A1001A82H"
DE_PHYSICAL_CONTROL_CODE = "10Y1001A1001A83F"

WIND_SOLAR_COLS = ["Wind Onshore", "Wind Offshore", "Solar"]


def _parse_utc(ts: str) -> pd.Timestamp:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return pd.Timestamp(dt).tz_convert("UTC")


def _month_ranges(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ranges = []
    cur = start
    while cur < end:
        nxt = (cur + pd.offsets.MonthBegin(1)).normalize()
        if nxt <= cur:
            nxt = cur + pd.DateOffset(months=1)
        ranges.append((cur, min(nxt, end)))
        cur = nxt
    return ranges


def _chunk_ranges(start: pd.Timestamp, end: pd.Timestamp, months: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if months <= 1:
        return _month_ranges(start, end)
    ranges = []
    cur = start
    while cur < end:
        nxt = (cur + pd.DateOffset(months=months)).normalize()
        if nxt <= cur:
            nxt = cur + pd.DateOffset(months=months)
        ranges.append((cur, min(nxt, end)))
        cur = nxt
    return ranges


def _retry(func, attempts: int = 3, sleep_s: float = 2.0):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # pragma: no cover - network errors
            last_exc = exc
            LOGGER.warning("Attempt %s failed: %s", attempt, exc)
            if attempt < attempts:
                time.sleep(sleep_s)
    raise RuntimeError(f"Failed after {attempts} attempts") from last_exc


def _ensure_utc_index(df: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def _resample_hourly(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    obj = _ensure_utc_index(obj)
    return obj.resample("1h").mean()


def _select_wind_solar(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols = [c for c in WIND_SOLAR_COLS if c in df.columns]
    if not cols:
        return df.iloc[0:0]
    return df[cols]


def _select_wind_solar_any(df: pd.DataFrame) -> pd.DataFrame:
    """Select wind/solar columns from flat or MultiIndex columns."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        keep = [c for c in df.columns if c[0] in WIND_SOLAR_COLS]
        return df[keep] if keep else df.iloc[0:0]
    return _select_wind_solar(df)


def _select_generation_actuals(df: pd.DataFrame) -> pd.DataFrame:
    """Select wind/solar actuals from entsoe-py generation response."""
    if df is None or df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        want = [("Wind Onshore", "Actual Aggregated"),
                ("Wind Offshore", "Actual Aggregated"),
                ("Solar", "Actual Aggregated")]
        cols = [c for c in want if c in df.columns]
        if not cols:
            # Fallback if only consumption is available.
            want = [("Wind Onshore", "Actual Consumption"),
                    ("Wind Offshore", "Actual Consumption"),
                    ("Solar", "Actual Consumption")]
            cols = [c for c in want if c in df.columns]
        return df[cols] if cols else df.iloc[0:0]

    # Flattened parquet columns often stringify tuples like "('Wind Onshore', 'Actual Aggregated')".
    def _pick(label: str, kind: str) -> str | None:
        key = f"('{label}', '{kind}')"
        return key if key in df.columns else None

    cols = [
        _pick("Wind Onshore", "Actual Aggregated"),
        _pick("Wind Offshore", "Actual Aggregated"),
        _pick("Solar", "Actual Aggregated"),
    ]
    cols = [c for c in cols if c is not None]
    if not cols:
        cols = [
            _pick("Wind Onshore", "Actual Consumption"),
            _pick("Wind Offshore", "Actual Consumption"),
            _pick("Solar", "Actual Consumption"),
        ]
        cols = [c for c in cols if c is not None]
    return df[cols] if cols else df.iloc[0:0]


def _rename_actual_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Wind Offshore": "wind_offshore_actual_entsoe",
        "Wind Onshore": "wind_onshore_actual_entsoe",
        "Solar": "solar_actual_entsoe",
    }
    if isinstance(df.columns, pd.MultiIndex):
        cols = {}
        for (psr, kind) in df.columns:
            if psr in mapping:
                cols[(psr, kind)] = mapping[psr]
        return df.rename(columns=cols)

    # Stringified tuple columns.
    str_map = {
        "('Wind Offshore', 'Actual Aggregated')": "wind_offshore_actual_entsoe",
        "('Wind Onshore', 'Actual Aggregated')": "wind_onshore_actual_entsoe",
        "('Solar', 'Actual Aggregated')": "solar_actual_entsoe",
        "('Wind Offshore', 'Actual Consumption')": "wind_offshore_actual_entsoe",
        "('Wind Onshore', 'Actual Consumption')": "wind_onshore_actual_entsoe",
        "('Solar', 'Actual Consumption')": "solar_actual_entsoe",
    }
    cols = {c: str_map[c] for c in df.columns if c in str_map}
    if cols:
        return df.rename(columns=cols)

    # Fallback: match on string representation to catch variant column names.
    fallback = {}
    for c in df.columns:
        s = str(c)
        if "Actual" not in s:
            continue
        if "Wind Onshore" in s:
            fallback[c] = "wind_onshore_actual_entsoe"
        elif "Wind Offshore" in s:
            fallback[c] = "wind_offshore_actual_entsoe"
        elif "Solar" in s:
            fallback[c] = "solar_actual_entsoe"
    if fallback:
        return df.rename(columns=fallback)

    cols = {c: mapping[c] for c in df.columns if c in mapping}
    return df.rename(columns=cols)


def _rename_forecast_cols(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    mapping = {
        "Wind Offshore": f"wind_offshore_forecast_{suffix}_entsoe",
        "Wind Onshore": f"wind_onshore_forecast_{suffix}_entsoe",
        "Solar": f"solar_forecast_{suffix}_entsoe",
    }
    cols = {c: mapping[c] for c in df.columns if c in mapping}
    return df.rename(columns=cols)


def _fetch_actuals(
    client: EntsoePandasClient, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame | None:
    try:
        actuals = _retry(lambda: client.query_generation(DE_PHYSICAL_CONTROL_CODE, start=start, end=end))
    except Exception as exc:  # pragma: no cover - network errors
        LOGGER.warning("Actual generation failed: %s", exc)
        return None

    if actuals is None or len(actuals) == 0:
        return None

    actuals = _ensure_utc_index(actuals)
    actuals = _select_generation_actuals(actuals)
    actuals = _rename_actual_cols(actuals)
    return _resample_hourly(actuals)


def _fetch_forecast(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    process_type: str,
    suffix: str,
) -> pd.DataFrame | None:
    try:
        forecast = _retry(
            lambda: client.query_wind_and_solar_forecast(
                DE_PHYSICAL_CONTROL_CODE, start=start, end=end, process_type=process_type
            )
        )
    except Exception as exc:  # pragma: no cover - network errors
        LOGGER.warning("Wind/solar forecast %s failed: %s", process_type, exc)
        return None

    if forecast is None or len(forecast) == 0:
        return None

    forecast = _ensure_utc_index(forecast)
    forecast = _select_wind_solar(forecast)
    forecast = _rename_forecast_cols(forecast, suffix)
    return _resample_hourly(forecast)


def _fetch_capacity(
    client: EntsoePandasClient, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame | None:
    try:
        capacity = _retry(lambda: client.query_installed_generation_capacity(
            DE_PHYSICAL_CONTROL_CODE, start=start, end=end
        ))
    except Exception as exc:  # pragma: no cover - network errors
        LOGGER.warning("Installed capacity fetch failed: %s", exc)
        return None

    if capacity is None or len(capacity) == 0:
        return None

    capacity = _ensure_utc_index(capacity)
    capacity = _select_wind_solar_any(capacity)
    if capacity is None or capacity.empty:
        LOGGER.warning("Installed capacity response has no Wind/Solar columns.")
        return None

    if isinstance(capacity.columns, pd.MultiIndex):
        cap_map = {}
        for c in capacity.columns:
            if c[0] == "Wind Onshore":
                cap_map[c] = "wind_onshore_capacity_entsoe"
            elif c[0] == "Wind Offshore":
                cap_map[c] = "wind_offshore_capacity_entsoe"
            elif c[0] == "Solar":
                cap_map[c] = "solar_capacity_entsoe"
        capacity = capacity.rename(columns=cap_map)
    else:
        capacity = capacity.rename(
            columns={
                "Wind Onshore": "wind_onshore_capacity_entsoe",
                "Wind Offshore": "wind_offshore_capacity_entsoe",
                "Solar": "solar_capacity_entsoe",
            }
        )

    # Capacity is sparse (often annual/monthly). Reindex to full hourly chunk and forward-fill.
    hourly_idx = pd.date_range(start=start, end=end, freq="1h", tz="UTC", inclusive="left")
    capacity = capacity.sort_index().reindex(hourly_idx).ffill()
    # Backfill head if first observed capacity point is after chunk start.
    capacity = capacity.bfill()
    return capacity


def fetch_chunk(client: EntsoePandasClient, start: pd.Timestamp, end: pd.Timestamp) -> pl.DataFrame:
    LOGGER.info("Fetching %s to %s", start, end)

    actuals = _fetch_actuals(client, start, end)

    da = _fetch_forecast(client, start, end, process_type="A01", suffix="da")

    id_forecast = _fetch_forecast(client, start, end, process_type="A18", suffix="id")
    if id_forecast is None:
        LOGGER.warning("A18 returned empty; trying A40 for intraday/current forecasts.")
        id_forecast = _fetch_forecast(client, start, end, process_type="A40", suffix="id")

    capacity = _fetch_capacity(client, start, end)

    frames = [df for df in (actuals, da, id_forecast, capacity) if df is not None and len(df) > 0]
    if frames:
        df = pd.concat(frames, axis=1, join="outer", sort=False)
        # Final cleanup: enforce actual column names in case tuple-like names survived.
        rename_map = {
            "('Wind Onshore', 'Actual Aggregated')": "wind_onshore_actual_entsoe",
            "('Wind Offshore', 'Actual Aggregated')": "wind_offshore_actual_entsoe",
            "('Solar', 'Actual Aggregated')": "solar_actual_entsoe",
            "('Wind Onshore', 'Actual Consumption')": "wind_onshore_actual_entsoe",
            "('Wind Offshore', 'Actual Consumption')": "wind_offshore_actual_entsoe",
            "('Solar', 'Actual Consumption')": "solar_actual_entsoe",
        }
        df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})
        # Handle tuple columns directly if present.
        tuple_map = {}
        for c in df.columns:
            if isinstance(c, tuple) and len(c) >= 2 and "Actual" in c[1]:
                if "Wind Onshore" in c[0]:
                    tuple_map[c] = "wind_onshore_actual_entsoe"
                elif "Wind Offshore" in c[0]:
                    tuple_map[c] = "wind_offshore_actual_entsoe"
                elif "Solar" in c[0]:
                    tuple_map[c] = "solar_actual_entsoe"
        if tuple_map:
            df = df.rename(columns=tuple_map)
    else:
        idx = pd.date_range(start=start, end=end, freq="1h", tz="UTC", inclusive="left")
        df = pd.DataFrame(index=idx)

    df.index = df.index.tz_convert("UTC")
    df = df.sort_index()

    pl_df = pl.from_pandas(df.reset_index()).rename({"index": "timestamp_utc"})
    return pl_df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fetch ENTSO-E wind/solar actuals + forecasts (DE-LU).")
    parser.add_argument("--start", required=True, help="Start ISO8601 (UTC).")
    parser.add_argument("--end", required=True, help="End ISO8601 (UTC).")
    parser.add_argument("--out", default="data/raw/entsoe.parquet", help="Output parquet path.")
    parser.add_argument("--chunk-months", type=int, default=3, help="Chunk size in months (default: 3).")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers (default: 3).")
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv(dotenv_path=Path(".env"))

    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ENTSOE_API_KEY environment variable")

    start = _parse_utc(args.start)
    end = _parse_utc(args.end)
    if end <= start:
        raise ValueError("--end must be after --start")

    client = EntsoePandasClient(api_key=api_key)

    ranges = _chunk_ranges(start, end, args.chunk_months)
    frames = []
    if args.workers <= 1:
        for s, e in ranges:
            frames.append(fetch_chunk(client, s, e))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(fetch_chunk, client, s, e): (s, e) for s, e in ranges}
            for fut in as_completed(futures):
                frames.append(fut.result())

    merged = (
        pl.concat(frames, how="diagonal")
        .unique(subset=["timestamp_utc"], keep="last")
        .sort("timestamp_utc")
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(out_path, compression="zstd")
    LOGGER.info("Wrote %s rows to %s", len(merged), out_path)


if __name__ == "__main__":
    main()
