"""Fetch ENTSO-E load data and generation outages via entsoe-py (hourly output).

Usage:
    python -m energy_trading.ingestion.fetch_entsoe \
        --start 2020-12-01T00:00:00Z --end 2026-01-01T02:00:00Z \
        --out data/raw/entsoe.parquet

Outputs:
    - entsoe.parquet with hourly UTC timestamps and columns:
        load_actual, load_forecast_da,
        outage_baseload_mw, outage_hard_coal_mw, outage_gas_mw
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from entsoe import EntsoePandasClient

LOGGER = logging.getLogger(__name__)
DEFAULT_COUNTRY = "DE_LU"


def _load_env_fallback() -> None:
    """Load ENTSOE_API_KEY from local .env if present."""
    if os.getenv("ENTSOE_API_KEY"):
        return
    candidates = [
        Path(__file__).resolve().parents[3] / ".env",  # repo root .env
        Path(__file__).resolve().parents[2] / ".env",  # src/.env (legacy)
        Path(__file__).resolve().parents[1] / ".env",  # package/.env (legacy)
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "ENTSOE_API_KEY":
                os.environ.setdefault("ENTSOE_API_KEY", value.strip().strip('"').strip("'"))
                return


def _to_utc(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    idx = data.index
    if idx.tz is None:
        data = data.tz_localize("Europe/Berlin", ambiguous="infer", nonexistent="shift_forward")
    return data.tz_convert("UTC")


def _safe_fetch(name: str, func):
    try:
        series = func()
        if series is None or len(series) == 0:
            LOGGER.warning("%s returned no data.", name)
            return None
        return series
    except Exception as exc:
        LOGGER.warning("%s fetch failed: %s", name, exc)
        return None


def _to_series(data: pd.Series | pd.DataFrame, name: str) -> pd.Series | None:
    if data is None or len(data) == 0:
        return None
    data = _to_utc(data)
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            data = data.iloc[:, 0]
        else:
            data = data.sum(axis=1)
    return data.rename(name)


def _outages_to_hourly(events: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    baseload = {"Nuclear", "Fossil Brown coal/Lignite"}
    hard_coal = {"Fossil Hard coal"}
    gas = {"Fossil Gas"}

    ts = idx.values
    base_vals = np.zeros(len(idx), dtype=float)
    coal_vals = np.zeros(len(idx), dtype=float)
    gas_vals = np.zeros(len(idx), dtype=float)

    mapped_baseload = 0
    mapped_hard_coal = 0
    mapped_gas = 0

    for _, row in events.iterrows():
        evt_start = max(row["start"], idx[0])
        evt_end = min(row["end"], idx[-1])
        if evt_start >= evt_end:
            continue
        i_start = np.searchsorted(ts, evt_start.to_datetime64(), side="left")
        i_end = np.searchsorted(ts, evt_end.to_datetime64(), side="left")
        if i_start >= i_end:
            continue
        fuel = row["plant_type"]
        if fuel in baseload:
            base_vals[i_start:i_end] += row["missing_mw"]
            mapped_baseload += 1
        elif fuel in hard_coal:
            coal_vals[i_start:i_end] += row["missing_mw"]
            mapped_hard_coal += 1
        elif fuel in gas:
            gas_vals[i_start:i_end] += row["missing_mw"]
            mapped_gas += 1

    LOGGER.info("Mapped %s events to Baseload outages.", mapped_baseload)
    LOGGER.info("Mapped %s events to Hard Coal outages.", mapped_hard_coal)
    LOGGER.info("Mapped %s events to Gas outages.", mapped_gas)

    return pd.DataFrame(
        {
            "outage_baseload_mw": base_vals,
            "outage_hard_coal_mw": coal_vals,
            "outage_gas_mw": gas_vals,
        },
        index=idx,
    )


def fetch_outages_as_timeseries(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    country: str,
) -> pd.DataFrame:
    """Fetch generation outages and convert to hourly time series by fuel group."""
    outages_raw = _safe_fetch(
        "generation_outages",
        lambda: client.query_unavailability_of_generation_units(country, start=start, end=end, docstatus="A05"),
    )
    if outages_raw is None or len(outages_raw) == 0:
        return pd.DataFrame()

    df = outages_raw.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index(drop=True)

    cols = {c.lower(): c for c in df.columns}
    start_col = cols.get("start")
    end_col = cols.get("end")
    avail_col = cols.get("avail_qty")
    nominal_col = cols.get("nominal_power")
    psr_col = cols.get("plant_type")

    if not (start_col and end_col and avail_col and nominal_col and psr_col):
        LOGGER.warning("Outage data missing expected columns; got: %s", list(df.columns))
        return pd.DataFrame()

    events = df[[start_col, end_col, avail_col, nominal_col, psr_col]].copy()
    events["start"] = pd.to_datetime(events[start_col], errors="coerce").dt.tz_convert("UTC")
    events["end"] = pd.to_datetime(events[end_col], errors="coerce").dt.tz_convert("UTC")
    events["avail_qty"] = pd.to_numeric(events[avail_col], errors="coerce").astype(float)
    events["nominal_power"] = pd.to_numeric(events[nominal_col], errors="coerce").astype(float)
    events["plant_type"] = events[psr_col]
    events = events.dropna(subset=["start", "end", "avail_qty", "nominal_power", "plant_type"])

    events["missing_mw"] = (events["nominal_power"] - events["avail_qty"]).clip(lower=0.0)

    idx = pd.date_range(start=start, end=end, freq="1h", tz="UTC", inclusive="left")
    return _outages_to_hourly(events, idx)


def fetch_entsoe(start: datetime, end: datetime, country: str) -> pd.DataFrame:
    _load_env_fallback()
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError("ENTSOE_API_KEY is not set in the environment.")

    client = EntsoePandasClient(api_key=api_key)
    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    end_ts = pd.Timestamp(end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")

    load_actual = _safe_fetch(
        "load_actual",
        lambda: client.query_load(country, start=start_ts, end=end_ts),
    )
    load_actual = _to_series(load_actual, "load_actual")

    load_forecast_da = _safe_fetch(
        "load_forecast_da",
        lambda: client.query_load_forecast(country, start=start_ts, end=end_ts, process_type="A01"),
    )
    load_forecast_da = _to_series(load_forecast_da, "load_forecast_da")

    series = [s for s in [load_actual, load_forecast_da] if s is not None and len(s) > 0]
    if not series:
        raise RuntimeError("No ENTSO-E load data fetched.")

    df_load = pd.concat(series, axis=1).sort_index()
    # Force hourly resolution
    df_load = df_load.resample("1h").mean()
    # Fix known 24h gap with linear interpolation
    if "load_forecast_da" in df_load.columns:
        df_load["load_forecast_da"] = df_load["load_forecast_da"].interpolate(method="linear", limit=24)

    outages = fetch_outages_as_timeseries(client, start_ts, end_ts, country)

    df = df_load.copy()
    if not outages.empty:
        df = df.join(outages, how="left")
        for col in ["outage_baseload_mw", "outage_hard_coal_mw", "outage_gas_mw"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    df = df.resample("1h").mean()
    df = df.reset_index().rename(columns={"index": "timestamp_utc"})
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fetch ENTSO-E load data and outages via entsoe-py (hourly).")
    parser.add_argument("--start", required=True, help="Start ISO8601 (UTC, inclusive).")
    parser.add_argument("--end", required=True, help="End ISO8601 (UTC, inclusive).")
    parser.add_argument("--country", default=DEFAULT_COUNTRY, help="ENTSO-E country/bidding zone code (default DE_LU).")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[3] / "data" / "raw" / "entsoe.parquet"),
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

    df = fetch_entsoe(start_dt, end_dt, args.country)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    LOGGER.info("Wrote %s rows to %s", len(df), out_path)


if __name__ == "__main__":
    main()
