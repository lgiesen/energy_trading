"""Fetch ENTSO-E wind/solar actuals and forecasts for DE-LU using entsoe-py.

Usage:
    ./.venv/bin/python -m energy_trading.ingestion.fetch_entsoe \
        --start 2020-11-30T23:00:00Z --end 2025-12-31T23:00:00Z \
        --out data/raw/entsoe.parquet

Outputs:
    - entsoe.parquet with hourly UTC timestamps and:
        timestamp_utc,
        wind_onshore_actual_entsoe, wind_offshore_actual_entsoe, solar_actual_entsoe,
        wind_onshore_forecast_da_entsoe, wind_offshore_forecast_da_entsoe, solar_forecast_da_entsoe,
        wind_onshore_forecast_id_entsoe, wind_offshore_forecast_id_entsoe, solar_forecast_id_entsoe,
        wind_onshore_capacity_entsoe, wind_offshore_capacity_entsoe,
        afrr_cbmp_pos_eur_mwh, afrr_cbmp_neg_eur_mwh
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
# Legacy alias used by older notebooks.
BIDDING_ZONE = DE_LU_BIDDING_ZONE_CODE

WIND_SOLAR_COLS = ["Wind Onshore", "Wind Offshore", "Solar"]
PSR_WIND_ONSHORE = "B19"
PSR_WIND_OFFSHORE = "B18"


def _parse_utc(ts: str) -> pd.Timestamp:
    # Accept ISO8601 and compact forms used in legacy notebooks (YYYYMMDDHHMM).
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        dt = pd.to_datetime(ts, utc=True, format="%Y%m%d%H%M", errors="raise").to_pydatetime()
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


def _find_directional_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Detect positive/upward and negative/downward price columns."""
    if df is None or df.empty:
        return None, None
    lower_map = {c: str(c).lower() for c in df.columns}

    def _pick(keywords: tuple[str, ...]) -> str | None:
        for c, lc in lower_map.items():
            if all(k in lc for k in keywords):
                return c
        for c, lc in lower_map.items():
            if any(k in lc for k in keywords):
                return c
        return None

    pos = _pick(("positive",)) or _pick(("upward",)) or _pick(("(+)",)) or _pick(("pos",))
    neg = _pick(("negative",)) or _pick(("downward",)) or _pick(("(-)",)) or _pick(("neg",))
    return pos, neg


def _empty_afrr_cbmp() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp_utc")
    return pd.DataFrame(
        {
            "afrr_cbmp_pos_eur_mwh": pd.Series(dtype="float64"),
            "afrr_cbmp_neg_eur_mwh": pd.Series(dtype="float64"),
        },
        index=idx,
    )


def fetch_afrr_cbmp(
    client: EntsoePandasClient, start_dt: pd.Timestamp, end_dt: pd.Timestamp
) -> pd.DataFrame:
    """Fetch aFRR CBMP (cross-border marginal prices) for DE-LU control area.

    Notes:
    - ENTSO-E library versions differ: some provide `query_balancing_prices`,
      others provide `query_activated_balancing_energy_prices`.
    - Data may be unavailable for early periods. In that case return an empty
      frame with stable schema so joins remain safe.
    - Raw data may be 15-minute; we aggregate to hourly by mean because price
      units are EUR/MWh.
    """
    raw: pd.DataFrame | pd.Series | None = None
    errors: list[str] = []

    # Preferred method if available in local entsoe-py.
    if hasattr(client, "query_balancing_prices"):
        try:
            raw = _retry(
                lambda: client.query_balancing_prices(
                    country_code=DE_PHYSICAL_CONTROL_CODE,
                    start=start_dt,
                    end=end_dt,
                    process_type="A16",
                )
            )
        except Exception as exc:  # pragma: no cover - network/API variation
            errors.append(f"query_balancing_prices: {exc}")

    # Fallback for currently installed entsoe-py versions.
    if raw is None:
        try:
            raw = _retry(
                lambda: client.query_activated_balancing_energy_prices(
                    country_code=DE_PHYSICAL_CONTROL_CODE,
                    start=start_dt,
                    end=end_dt,
                    process_type="A16",
                )
            )
        except Exception as exc:  # pragma: no cover - network/API variation
            errors.append(f"query_activated_balancing_energy_prices: {exc}")

    if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
        if errors:
            LOGGER.info("aFRR CBMP unavailable for %s -> %s (%s)", start_dt, end_dt, " | ".join(errors))
        return _empty_afrr_cbmp()

    if isinstance(raw, pd.Series):
        frame = raw.to_frame(name="afrr_cbmp_pos_eur_mwh")
        frame["afrr_cbmp_neg_eur_mwh"] = pd.NA
    else:
        frame = raw.copy()

    frame = _ensure_utc_index(frame)
    frame.index.name = "timestamp_utc"

    pos_col, neg_col = _find_directional_columns(frame)
    if pos_col is None and neg_col is None:
        # If only one numeric series is returned, map to pos and keep neg null.
        num_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        if len(num_cols) == 1:
            frame = frame[[num_cols[0]]].rename(columns={num_cols[0]: "afrr_cbmp_pos_eur_mwh"})
            frame["afrr_cbmp_neg_eur_mwh"] = pd.NA
        else:
            LOGGER.info("aFRR CBMP columns not recognized for %s -> %s; returning empty schema.", start_dt, end_dt)
            return _empty_afrr_cbmp()
    else:
        out = pd.DataFrame(index=frame.index)
        out["afrr_cbmp_pos_eur_mwh"] = frame[pos_col] if pos_col is not None else pd.NA
        out["afrr_cbmp_neg_eur_mwh"] = frame[neg_col] if neg_col is not None else pd.NA
        frame = out

    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = _resample_hourly(frame)  # Prices: hourly mean from 15-min EUR/MWh.
    frame.index = frame.index.tz_convert("UTC")
    frame.index.name = "timestamp_utc"
    return frame


def _fetch_capacity_yearly(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch sparse installed capacities by year and return UTC-indexed time series.

    ENTSO-E installed generation capacity is typically sparse (e.g., annual or monthly updates),
    so we query per year and later forward-fill onto the hourly grid.
    """

    def _fetch_psr(
        psr_type: str,
        year_start: pd.Timestamp,
        year_end: pd.Timestamp,
        label: str,
    ) -> pd.Series | None:
        try:
            out = _retry(
                lambda: client.query_installed_generation_capacity(
                    DE_PHYSICAL_CONTROL_CODE,
                    start=year_start,
                    end=year_end,
                    psr_type=psr_type,
                )
            )
        except Exception as exc:  # pragma: no cover - network errors
            LOGGER.warning("Installed capacity %s failed for %s: %s", label, year_start.year, exc)
            return None
        if out is None or len(out) == 0:
            LOGGER.warning("Installed capacity %s empty for %s.", label, year_start.year)
            return None
        if isinstance(out, pd.DataFrame):
            # Defensive fallback: reduce dataframe responses to one numeric series.
            num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
            if not num_cols:
                return None
            series = out[num_cols[0]]
        else:
            series = out
        series = pd.to_numeric(series, errors="coerce")
        series = _ensure_utc_index(series)
        return series

    year_frames: list[pd.DataFrame] = []
    for year in range(start.year, end.year + 1):
        year_start = max(start, pd.Timestamp(year=year, month=1, day=1, tz="UTC"))
        year_end = min(end, pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC"))
        if year_end <= year_start:
            continue

        onshore = _fetch_psr(PSR_WIND_ONSHORE, year_start, year_end, "wind_onshore")
        offshore = _fetch_psr(PSR_WIND_OFFSHORE, year_start, year_end, "wind_offshore")
        if onshore is None and offshore is None:
            continue

        year_df = pd.DataFrame(index=pd.Index([], dtype="datetime64[ns, UTC]"))
        if onshore is not None:
            year_df = year_df.join(
                onshore.rename("wind_onshore_capacity_entsoe"), how="outer"
            )
        if offshore is not None:
            year_df = year_df.join(
                offshore.rename("wind_offshore_capacity_entsoe"), how="outer"
            )
        year_frames.append(year_df)

    if not year_frames:
        return pd.DataFrame()

    cap = pd.concat(year_frames, axis=0).sort_index()
    cap = cap[~cap.index.duplicated(keep="last")]
    return cap


def fetch_chunk(client: EntsoePandasClient, start: pd.Timestamp, end: pd.Timestamp) -> pl.DataFrame:
    LOGGER.info("Fetching %s to %s", start, end)

    actuals = _fetch_actuals(client, start, end)

    da = _fetch_forecast(client, start, end, process_type="A01", suffix="da")

    id_forecast = _fetch_forecast(client, start, end, process_type="A18", suffix="id")
    if id_forecast is None:
        LOGGER.warning("A18 returned empty; trying A40 for intraday/current forecasts.")
        id_forecast = _fetch_forecast(client, start, end, process_type="A40", suffix="id")
    afrr_cbmp = fetch_afrr_cbmp(client, start, end)

    frames = [df for df in (actuals, da, id_forecast, afrr_cbmp) if df is not None and len(df) > 0]
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

    for col in ("afrr_cbmp_pos_eur_mwh", "afrr_cbmp_neg_eur_mwh"):
        if col not in df.columns:
            df[col] = pd.NA

    df.index = df.index.tz_convert("UTC")
    df = df.sort_index()

    pl_df = pl.from_pandas(df.reset_index()).rename({"index": "timestamp_utc"})
    return pl_df


def fetch_and_merge(
    start: str,
    end: str,
    bidding_zone_or_out: str | None = None,
    process_year_ahead: str | None = None,
    out: str | None = None,
    chunk_months: int = 3,
    workers: int = 3,
    token: str | None = None,
    timeout: int | None = None,
    chunk_days: int | None = None,
) -> pl.DataFrame:
    """Backward-compatible notebook helper.

    Notes:
    - `timeout` and `chunk_days` are accepted for compatibility and ignored.
    - `process_year_ahead` is accepted for compatibility and ignored.
    - `start`/`end` accept ISO8601 and compact `YYYYMMDDHHMM`.
    - Third positional argument can be either legacy bidding-zone code or output path.
    """
    _ = timeout
    _ = chunk_days
    _ = process_year_ahead

    bidding_zone = DE_LU_BIDDING_ZONE_CODE
    if bidding_zone_or_out:
        if str(bidding_zone_or_out).startswith("10Y"):
            bidding_zone = str(bidding_zone_or_out)
        elif out is None:
            out = str(bidding_zone_or_out)

    if load_dotenv is not None:
        load_dotenv(dotenv_path=Path(".env"))

    api_key = token or os.getenv("ENTSOE_API_TOKEN") or os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ENTSOE_API_TOKEN (or ENTSOE_API_KEY) environment variable")

    start_ts = _parse_utc(start)
    end_ts = _parse_utc(end)
    if end_ts <= start_ts:
        raise ValueError("end must be after start")
    query_end = end_ts + pd.Timedelta(hours=1)

    _ = bidding_zone  # kept for interface compatibility; current implementation uses fixed DE-LU mappings.
    client = EntsoePandasClient(api_key=api_key)
    ranges = _chunk_ranges(start_ts, query_end, chunk_months)
    frames: list[pl.DataFrame] = []

    if workers <= 1:
        for s, e in ranges:
            frames.append(fetch_chunk(client, s, e))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(fetch_chunk, client, s, e): (s, e) for s, e in ranges}
            for fut in as_completed(futures):
                frames.append(fut.result())

    merged = (
        pl.concat(frames, how="diagonal")
        .unique(subset=["timestamp_utc"], keep="last")
        .sort("timestamp_utc")
    )

    cap_sparse = _fetch_capacity_yearly(client, start_ts, query_end)
    if not cap_sparse.empty:
        hourly = merged.to_pandas().set_index("timestamp_utc").sort_index()
        hourly = hourly.join(cap_sparse, how="left")
        for col in ("wind_onshore_capacity_entsoe", "wind_offshore_capacity_entsoe"):
            if col in hourly.columns:
                hourly[col] = hourly[col].ffill()
        merged = pl.from_pandas(hourly.reset_index())

    merged = merged.filter(pl.col("timestamp_utc") <= pl.lit(end_ts))
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.write_parquet(out_path, compression="zstd")
    return merged


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

    api_key = os.getenv("ENTSOE_API_TOKEN") or os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ENTSOE_API_TOKEN (or ENTSOE_API_KEY) environment variable")

    start = _parse_utc(args.start)
    end = _parse_utc(args.end)
    if end <= start:
        raise ValueError("--end must be after --start")
    # ENTSO-E API windows are end-exclusive; query one extra hour and clip back.
    query_end = end + pd.Timedelta(hours=1)

    client = EntsoePandasClient(api_key=api_key)

    ranges = _chunk_ranges(start, query_end, args.chunk_months)
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

    # Installed capacities are sparse; fetch yearly and forward-fill to hourly grid.
    cap_sparse = _fetch_capacity_yearly(client, start, query_end)
    if cap_sparse.empty:
        LOGGER.warning("Installed generation capacity endpoint returned no usable rows.")
    else:
        hourly = merged.to_pandas().set_index("timestamp_utc").sort_index()
        hourly = hourly.join(cap_sparse, how="left")
        for col in ("wind_onshore_capacity_entsoe", "wind_offshore_capacity_entsoe"):
            if col in hourly.columns:
                hourly[col] = hourly[col].ffill()
                LOGGER.info("Non-null %s: %s", col, int(hourly[col].notna().sum()))
        merged = pl.from_pandas(hourly.reset_index())

    merged = merged.filter(pl.col("timestamp_utc") <= pl.lit(end))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(out_path, compression="zstd")
    LOGGER.info("Wrote %s rows to %s", len(merged), out_path)


if __name__ == "__main__":
    main()
