"""Fetch ENTSO-E load data and generation outages via entsoe-py.

Usage:
    python -m ingestion.fetch_entsoe \
        --start 2022-01-01 --end 2025-12-31 \
        --out data/entsoe.parquet

Outputs:
    - entsoe.parquet with hourly UTC timestamps and columns:
        load_actual, load_forecast_da,
        outage_baseload_mw, outage_hard_coal_mw, outage_gas_mw
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient

LOGGER = logging.getLogger(__name__)
DEFAULT_COUNTRY = "DE_LU"


def _load_env_fallback() -> None:
    """Load ENTSOE_API_KEY from local .env if present."""
    if os.getenv("ENTSOE_API_KEY"):
        return
    candidates = [
        Path(__file__).resolve().parents[1] / ".env",  # /energy_trading/.env
        Path(__file__).resolve().parents[2] / ".env",  # repo root .env
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
    psr_col = cols.get("plant_type") or cols.get("production_resource_psr_name")

    if not (start_col and end_col and avail_col and nominal_col and psr_col):
        LOGGER.warning("Outage data missing expected columns; got: %s", list(df.columns))
        return pd.DataFrame()

    events = df[[start_col, end_col, avail_col, nominal_col, psr_col]].copy()
    events[start_col] = pd.to_datetime(events[start_col], errors="coerce").dt.tz_convert("UTC")
    events[end_col] = pd.to_datetime(events[end_col], errors="coerce").dt.tz_convert("UTC")
    events[avail_col] = pd.to_numeric(events[avail_col], errors="coerce").astype(float)
    events[nominal_col] = pd.to_numeric(events[nominal_col], errors="coerce").astype(float)
    events = events.dropna(subset=[start_col, end_col, avail_col, nominal_col, psr_col])

    events["missing_mw"] = (events[nominal_col] - events[avail_col]).clip(lower=0.0)

    baseload = {"Nuclear", "Fossil Brown coal/Lignite"}
    hard_coal = {"Fossil Hard coal"}
    gas = {"Fossil Gas"}

    idx = pd.date_range(start=start, end=end, freq="1h", tz="UTC", inclusive="left")
    mapped_baseload = 0
    mapped_hard_coal = 0
    mapped_gas = 0
    out = pd.DataFrame(
        {
            "outage_baseload_mw": 0.0,
            "outage_hard_coal_mw": 0.0,
            "outage_gas_mw": 0.0,
        },
        index=idx,
    )

    for _, row in events.iterrows():
        evt_start = max(row[start_col], idx[0])
        evt_end = min(row[end_col], idx[-1])
        if evt_start >= evt_end:
            continue
        hours = pd.date_range(evt_start.floor("1h"), evt_end.floor("1h"), freq="1h", tz="UTC", inclusive="left")
        fuel = str(row[psr_col])
        if fuel in baseload:
            out.loc[hours, "outage_baseload_mw"] += float(row["missing_mw"])
            mapped_baseload += 1
        elif fuel in hard_coal:
            out.loc[hours, "outage_hard_coal_mw"] += float(row["missing_mw"])
            mapped_hard_coal += 1
        elif fuel in gas:
            out.loc[hours, "outage_gas_mw"] += float(row["missing_mw"])
            mapped_gas += 1

    LOGGER.info("Mapped %s events to Baseload outages.", mapped_baseload)
    LOGGER.info("Mapped %s events to Hard Coal outages.", mapped_hard_coal)
    LOGGER.info("Mapped %s events to Gas outages.", mapped_gas)

    return out


def fetch_entsoe(start: datetime, end: datetime, country: str) -> pd.DataFrame:
    _load_env_fallback()
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError("ENTSOE_API_KEY is not set in the environment.")

    client = EntsoePandasClient(api_key=api_key)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

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

    outages = fetch_outages_as_timeseries(client, start_ts, end_ts, country)

    series = [s for s in [load_actual, load_forecast_da] if s is not None and len(s) > 0]
    if not series:
        raise RuntimeError("No ENTSO-E load data fetched.")

    df = pd.concat(series, axis=1).sort_index()
    if not outages.empty:
        df = df.join(outages, how="left")
        for col in ["outage_baseload_mw", "outage_hard_coal_mw", "outage_gas_mw"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

    df = df.loc[(df.index >= start_ts) & (df.index < end_ts)]
    df = df.reset_index().rename(columns={"index": "timestamp"})
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fetch ENTSO-E load data and outages via entsoe-py.")
    parser.add_argument("--start", default="2022-01-01", help="Start date (UTC, inclusive).")
    parser.add_argument("--end", default="2025-12-31", help="End date (UTC, inclusive).")
    parser.add_argument("--country", default=DEFAULT_COUNTRY, help="ENTSO-E country/bidding zone code (default DE_LU).")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[2] / "data" / "entsoe.parquet"),
        help="Output parquet path.",
    )
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)

    df = fetch_entsoe(start_dt, end_dt, args.country)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    LOGGER.info("Wrote %s rows to %s", len(df), out_path)


if __name__ == "__main__":
    main()
