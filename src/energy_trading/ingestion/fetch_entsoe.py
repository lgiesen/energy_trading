"""Fetch ENTSO-E wind/solar actuals and forecasts for DE-LU using entsoe-py.

Usage:
    ./.venv/bin/python -m energy_trading.ingestion.fetch_entsoe \
        --start 2020-11-30T23:00:00Z --end 2026-03-01T02:00:00Z \
        --out data/raw/entsoe.parquet

Outputs:
    - entsoe.parquet with hourly UTC timestamps and:
        timestamp_utc,
        load_actual_entsoe, load_forecast_da_entsoe,
        wind_onshore_actual_entsoe, wind_offshore_actual_entsoe, solar_actual_entsoe,
        biomass_actual_entsoe, hydro_ror_actual_entsoe, hydro_reservoir_actual_entsoe, hydro_pumped_actual_entsoe,
        wind_onshore_forecast_da_entsoe, wind_offshore_forecast_da_entsoe, solar_forecast_da_entsoe,
        wind_onshore_forecast_id_entsoe, wind_offshore_forecast_id_entsoe, solar_forecast_id_entsoe,
        biomass_capacity_entsoe, hydro_pumped_capacity_entsoe, hydro_ror_capacity_entsoe,
        hydro_reservoir_capacity_entsoe, solar_capacity_entsoe,
        wind_onshore_capacity_entsoe, wind_offshore_capacity_entsoe
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
PSR_BIOMASS = "B01"
PSR_HYDRO_PUMPED = "B10"
PSR_HYDRO_ROR = "B11"
PSR_HYDRO_RESERVOIR = "B12"
PSR_SOLAR = "B16"

CAPACITY_PSR_MAP = {
    PSR_BIOMASS: ("Biomass", "biomass_capacity_entsoe"),
    PSR_HYDRO_PUMPED: ("Hydro Pumped Storage", "hydro_pumped_capacity_entsoe"),
    PSR_HYDRO_ROR: ("Hydro Run-of-river and poundage", "hydro_ror_capacity_entsoe"),
    PSR_HYDRO_RESERVOIR: ("Hydro Water Reservoir", "hydro_reservoir_capacity_entsoe"),
    PSR_SOLAR: ("Solar", "solar_capacity_entsoe"),
    PSR_WIND_OFFSHORE: ("Wind Offshore", "wind_offshore_capacity_entsoe"),
    PSR_WIND_ONSHORE: ("Wind Onshore", "wind_onshore_capacity_entsoe"),
}


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


DEFAULT_RETRY_ATTEMPTS = 4
DEFAULT_RETRY_SLEEP_S = 2.0


def _retry(func, attempts: int = DEFAULT_RETRY_ATTEMPTS, sleep_s: float = DEFAULT_RETRY_SLEEP_S):
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
        hydro_ror_labels = [
            "Hydro Run-of-River and poundage",
            "Hydro Run-of-river and poundage",
            "Hydro Run-of-River and pondage",
            "Hydro Run-of-river and pondage",
        ]
        want = [
            ("Wind Onshore", "Actual Aggregated"),
            ("Wind Offshore", "Actual Aggregated"),
            ("Solar", "Actual Aggregated"),
            ("Biomass", "Actual Aggregated"),
            ("Hydro Water Reservoir", "Actual Aggregated"),
            ("Hydro Pumped Storage", "Actual Aggregated"),
            ("Other storage", "Actual Aggregated"),
        ]
        want += [(lbl, "Actual Aggregated") for lbl in hydro_ror_labels]
        cols = [c for c in want if c in df.columns]
        if not cols:
            # Fallback if only consumption is available.
            want = [
                ("Wind Onshore", "Actual Consumption"),
                ("Wind Offshore", "Actual Consumption"),
                ("Solar", "Actual Consumption"),
                ("Biomass", "Actual Consumption"),
                ("Hydro Water Reservoir", "Actual Consumption"),
                ("Hydro Pumped Storage", "Actual Consumption"),
                ("Other storage", "Actual Consumption"),
            ]
            want += [(lbl, "Actual Consumption") for lbl in hydro_ror_labels]
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
        _pick("Biomass", "Actual Aggregated"),
        _pick("Hydro Run-of-River and poundage", "Actual Aggregated"),
        _pick("Hydro Water Reservoir", "Actual Aggregated"),
        _pick("Hydro Pumped Storage", "Actual Aggregated"),
        _pick("Other storage", "Actual Aggregated"),
    ]
    cols = [c for c in cols if c is not None]
    if not cols:
        cols = [
            _pick("Wind Onshore", "Actual Consumption"),
            _pick("Wind Offshore", "Actual Consumption"),
            _pick("Solar", "Actual Consumption"),
            _pick("Biomass", "Actual Consumption"),
            _pick("Hydro Run-of-River and poundage", "Actual Consumption"),
            _pick("Hydro Water Reservoir", "Actual Consumption"),
            _pick("Hydro Pumped Storage", "Actual Consumption"),
            _pick("Other storage", "Actual Consumption"),
        ]
        cols = [c for c in cols if c is not None]
    return df[cols] if cols else df.iloc[0:0]


def _rename_actual_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Wind Offshore": "wind_offshore_actual_entsoe",
        "Wind Onshore": "wind_onshore_actual_entsoe",
        "Solar": "solar_actual_entsoe",
        "Biomass": "biomass_actual_entsoe",
        "Hydro Run-of-River and poundage": "hydro_ror_actual_entsoe",
        "Hydro Water Reservoir": "hydro_reservoir_actual_entsoe",
        "Hydro Pumped Storage": "hydro_pumped_actual_entsoe",
        "Other storage": "hydro_pumped_actual_entsoe",
    }
    if isinstance(df.columns, pd.MultiIndex):
        # Pandas MultiIndex rename with tuple-keys can be unreliable across versions.
        # Build a flat single-level column index explicitly.
        new_cols = []
        for (psr, _kind) in df.columns:
            psr_str = str(psr)
            psr_low = psr_str.lower()
            if psr_str in mapping:
                new_cols.append(mapping[psr_str])
            elif "run-of-river" in psr_low and ("poundage" in psr_low or "pondage" in psr_low):
                new_cols.append("hydro_ror_actual_entsoe")
            else:
                new_cols.append(psr_str)
        out = df.copy()
        out.columns = new_cols
        return out

    # Stringified tuple columns.
    str_map = {
        "('Wind Offshore', 'Actual Aggregated')": "wind_offshore_actual_entsoe",
        "('Wind Onshore', 'Actual Aggregated')": "wind_onshore_actual_entsoe",
        "('Solar', 'Actual Aggregated')": "solar_actual_entsoe",
        "('Wind Offshore', 'Actual Consumption')": "wind_offshore_actual_entsoe",
        "('Wind Onshore', 'Actual Consumption')": "wind_onshore_actual_entsoe",
        "('Solar', 'Actual Consumption')": "solar_actual_entsoe",
        "('Biomass', 'Actual Aggregated')": "biomass_actual_entsoe",
        "('Biomass', 'Actual Consumption')": "biomass_actual_entsoe",
        "('Hydro Run-of-River and poundage', 'Actual Aggregated')": "hydro_ror_actual_entsoe",
        "('Hydro Run-of-River and poundage', 'Actual Consumption')": "hydro_ror_actual_entsoe",
        "('Hydro Water Reservoir', 'Actual Aggregated')": "hydro_reservoir_actual_entsoe",
        "('Hydro Water Reservoir', 'Actual Consumption')": "hydro_reservoir_actual_entsoe",
        "('Hydro Pumped Storage', 'Actual Aggregated')": "hydro_pumped_actual_entsoe",
        "('Hydro Pumped Storage', 'Actual Consumption')": "hydro_pumped_actual_entsoe",
        "('Other storage', 'Actual Aggregated')": "hydro_pumped_actual_entsoe",
        "('Other storage', 'Actual Consumption')": "hydro_pumped_actual_entsoe",
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
        s_low = s.lower()
        if "Wind Onshore" in s:
            fallback[c] = "wind_onshore_actual_entsoe"
        elif "Wind Offshore" in s:
            fallback[c] = "wind_offshore_actual_entsoe"
        elif "Solar" in s:
            fallback[c] = "solar_actual_entsoe"
        elif "Biomass" in s:
            fallback[c] = "biomass_actual_entsoe"
        elif "run-of-river" in s_low and ("poundage" in s_low or "pondage" in s_low):
            fallback[c] = "hydro_ror_actual_entsoe"
        elif "Hydro Water Reservoir" in s:
            fallback[c] = "hydro_reservoir_actual_entsoe"
        elif "Hydro Pumped Storage" in s or "Other storage" in s:
            fallback[c] = "hydro_pumped_actual_entsoe"
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
    target_cols = [
        "wind_onshore_actual_entsoe",
        "wind_offshore_actual_entsoe",
        "solar_actual_entsoe",
        "biomass_actual_entsoe",
        "hydro_ror_actual_entsoe",
        "hydro_reservoir_actual_entsoe",
        "hydro_pumped_actual_entsoe",
    ]
    area_candidates = [DE_LU_BIDDING_ZONE_CODE, DE_PHYSICAL_CONTROL_CODE]
    combined: pd.DataFrame | None = None
    for area in area_candidates:
        try:
            actuals = _retry(lambda: client.query_generation(area, start=start, end=end))
        except Exception as exc:  # pragma: no cover - network errors
            LOGGER.warning("Actual generation failed for area %s: %s", area, exc)
            continue

        if actuals is None or len(actuals) == 0:
            continue

        actuals = _ensure_utc_index(actuals)
        actuals = _select_generation_actuals(actuals)
        actuals = _rename_actual_cols(actuals)
        if actuals is None or len(actuals) == 0:
            continue
        actuals = _resample_hourly(actuals)

        if combined is None:
            combined = actuals
        else:
            combined = combined.combine_first(actuals)

        missing = [
            c
            for c in target_cols
            if c not in combined.columns or combined[c].notna().sum() == 0
        ]
        if not missing:
            break
        LOGGER.info(
            "Actual generation area %s merged; still missing columns: %s",
            area,
            missing,
        )
    return combined


def _fetch_forecast(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    process_type: str,
    suffix: str,
) -> pd.DataFrame | None:
    area_candidates = [DE_LU_BIDDING_ZONE_CODE, DE_PHYSICAL_CONTROL_CODE]
    for area in area_candidates:
        try:
            forecast = _retry(
                lambda: client.query_wind_and_solar_forecast(
                    area, start=start, end=end, process_type=process_type
                )
            )
        except Exception as exc:  # pragma: no cover - network errors
            LOGGER.warning(
                "Wind/solar forecast %s failed for area %s: %s",
                process_type,
                area,
                exc,
            )
            continue

        if forecast is None or len(forecast) == 0:
            continue

        forecast = _ensure_utc_index(forecast)
        forecast = _select_wind_solar(forecast)
        if forecast is None or len(forecast) == 0:
            continue
        forecast = _rename_forecast_cols(forecast, suffix)
        return _resample_hourly(forecast)
    return None


def _fetch_load_actual(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame | None:
    """Fetch ENTSO-E total actual load."""
    area_candidates = [DE_LU_BIDDING_ZONE_CODE, DE_PHYSICAL_CONTROL_CODE]
    for area in area_candidates:
        try:
            load = _retry(lambda: client.query_load(area, start=start, end=end))
        except Exception as exc:  # pragma: no cover - network/API variation
            LOGGER.warning("Load actual failed for area %s: %s", area, exc)
            continue
        if load is None or len(load) == 0:
            continue
        if isinstance(load, pd.DataFrame):
            num_cols = [c for c in load.columns if pd.api.types.is_numeric_dtype(load[c])]
            if not num_cols:
                continue
            load = load[num_cols[0]]
        load = pd.to_numeric(load, errors="coerce")
        load = _resample_hourly(load)
        return load.to_frame(name="load_actual_entsoe")
    return None


def _fetch_load_forecast_da(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame | None:
    """Fetch ENTSO-E day-ahead load forecast."""
    area_candidates = [DE_LU_BIDDING_ZONE_CODE, DE_PHYSICAL_CONTROL_CODE]
    # A01 = day-ahead, A31 = common fallback in ENTSO-E payloads.
    process_candidates = ["A01", "A31"]
    for area in area_candidates:
        for process_type in process_candidates:
            try:
                forecast = _retry(
                    lambda: client.query_load_forecast(
                        area,
                        start=start,
                        end=end,
                        process_type=process_type,
                    )
                )
            except TypeError:
                # Some entsoe-py versions do not accept process_type.
                try:
                    forecast = _retry(lambda: client.query_load_forecast(area, start=start, end=end))
                except Exception as exc:  # pragma: no cover - network/API variation
                    LOGGER.warning(
                        "Load forecast failed for area %s (process %s): %s",
                        area,
                        process_type,
                        exc,
                    )
                    continue
            except Exception as exc:  # pragma: no cover - network/API variation
                LOGGER.warning(
                    "Load forecast failed for area %s (process %s): %s",
                    area,
                    process_type,
                    exc,
                )
                continue

            if forecast is None or len(forecast) == 0:
                continue
            if isinstance(forecast, pd.DataFrame):
                num_cols = [c for c in forecast.columns if pd.api.types.is_numeric_dtype(forecast[c])]
                if not num_cols:
                    continue
                forecast = forecast[num_cols[0]]
            forecast = pd.to_numeric(forecast, errors="coerce")
            forecast = _resample_hourly(forecast)
            return forecast.to_frame(name="load_forecast_da_entsoe")
    return None


def _fetch_capacity_yearly(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch installed generation capacities from the DE-LU bidding zone by year.

    ENTSO-E 14.1.b capacity responses for Germany are published on bidding-zone
    level. We therefore query the full capacity table once per year for
    `DE_LU_BIDDING_ZONE_CODE`, extract the relevant PSR columns, and assign the
    values to a single yearly UTC timestamp (Jan 1st 00:00 UTC).
    """

    def _extract_capacity_value(
        out: pd.DataFrame | pd.Series,
        label: str,
    ) -> float | None:
        if out is None:
            return None

        if isinstance(out, pd.Series):
            numeric = pd.to_numeric(out, errors="coerce").dropna()
            return float(numeric.iloc[-1]) if not numeric.empty else None

        if not isinstance(out, pd.DataFrame) or out.empty:
            return None

        if label in out.columns:
            numeric = pd.to_numeric(out[label], errors="coerce").dropna()
            return float(numeric.iloc[-1]) if not numeric.empty else None

        for col in out.columns:
            if str(col).strip().lower() == label.strip().lower():
                numeric = pd.to_numeric(out[col], errors="coerce").dropna()
                return float(numeric.iloc[-1]) if not numeric.empty else None

        return None

    records: list[dict[str, float | pd.Timestamp]] = []
    for year in range(start.year, end.year + 1):
        year_start = max(start, pd.Timestamp(year=year, month=1, day=1, tz="UTC"))
        year_end = min(end, pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC"))
        if year_end <= year_start:
            continue

        try:
            out = _retry(
                lambda: client.query_installed_generation_capacity(
                    DE_LU_BIDDING_ZONE_CODE,
                    start=year_start,
                    end=year_end,
                )
            )
        except Exception as exc:  # pragma: no cover - network errors
            LOGGER.warning(
                "Installed generation capacity table failed for %s: %s",
                year,
                exc,
            )
            continue

        row: dict[str, float | pd.Timestamp] = {
            "timestamp_utc": pd.Timestamp(year=year, month=1, day=1, tz="UTC"),
        }
        found_any = False
        for _psr, (label, out_col) in CAPACITY_PSR_MAP.items():
            value = _extract_capacity_value(out, label)
            row[out_col] = value
            if value is not None:
                found_any = True

        if not found_any:
            LOGGER.warning(
                "Installed generation capacity table returned no usable rows for %s.",
                year,
            )
            continue
        records.append(row)

    if not records:
        return pd.DataFrame()

    cap = pd.DataFrame.from_records(records).set_index("timestamp_utc").sort_index()
    cap.index = pd.to_datetime(cap.index, utc=True)
    return cap


def _apply_capacity_ffill(
    merged: pl.DataFrame,
    cap_sparse: pd.DataFrame,
) -> pl.DataFrame:
    """Join sparse ENTSO-E capacity points onto hourly grid and forward-fill."""
    if cap_sparse.empty:
        return merged

    hourly = merged.to_pandas().set_index("timestamp_utc").sort_index()
    hourly = hourly.join(cap_sparse, how="left")

    capacity_cols = [
        out_col for _label, out_col in CAPACITY_PSR_MAP.values()
        if out_col in hourly.columns
    ]
    for col in capacity_cols:
        hourly[col] = pd.to_numeric(hourly[col], errors="coerce").ffill()
        LOGGER.info("Non-null %s: %s", col, int(hourly[col].notna().sum()))

    return pl.from_pandas(hourly.reset_index())


def fetch_chunk(client: EntsoePandasClient, start: pd.Timestamp, end: pd.Timestamp) -> pl.DataFrame:
    LOGGER.info("Fetching %s to %s", start, end)

    load_actual = _fetch_load_actual(client, start, end)
    load_forecast_da = _fetch_load_forecast_da(client, start, end)
    actuals = _fetch_actuals(client, start, end)

    da = _fetch_forecast(client, start, end, process_type="A01", suffix="da")

    id_forecast = _fetch_forecast(client, start, end, process_type="A18", suffix="id")
    if id_forecast is None:
        LOGGER.warning("A18 returned empty; trying A40 for intraday/current forecasts.")
        id_forecast = _fetch_forecast(client, start, end, process_type="A40", suffix="id")

    frames = [df for df in (load_actual, load_forecast_da, actuals, da, id_forecast) if df is not None and len(df) > 0]
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
            "('Biomass', 'Actual Aggregated')": "biomass_actual_entsoe",
            "('Biomass', 'Actual Consumption')": "biomass_actual_entsoe",
            "('Hydro Run-of-River and poundage', 'Actual Aggregated')": "hydro_ror_actual_entsoe",
            "('Hydro Run-of-River and poundage', 'Actual Consumption')": "hydro_ror_actual_entsoe",
            "('Hydro Water Reservoir', 'Actual Aggregated')": "hydro_reservoir_actual_entsoe",
            "('Hydro Water Reservoir', 'Actual Consumption')": "hydro_reservoir_actual_entsoe",
            "('Hydro Pumped Storage', 'Actual Aggregated')": "hydro_pumped_actual_entsoe",
            "('Hydro Pumped Storage', 'Actual Consumption')": "hydro_pumped_actual_entsoe",
            "('Other storage', 'Actual Aggregated')": "hydro_pumped_actual_entsoe",
            "('Other storage', 'Actual Consumption')": "hydro_pumped_actual_entsoe",
        }
        df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})
        # Handle tuple columns directly if present.
        tuple_map = {}
        for c in df.columns:
            if isinstance(c, tuple) and len(c) >= 2 and "Actual" in c[1]:
                psr = str(c[0])
                psr_low = psr.lower()
                if "Wind Onshore" in c[0]:
                    tuple_map[c] = "wind_onshore_actual_entsoe"
                elif "Wind Offshore" in c[0]:
                    tuple_map[c] = "wind_offshore_actual_entsoe"
                elif "Solar" in c[0]:
                    tuple_map[c] = "solar_actual_entsoe"
                elif "Biomass" in c[0]:
                    tuple_map[c] = "biomass_actual_entsoe"
                elif "run-of-river" in psr_low and ("poundage" in psr_low or "pondage" in psr_low):
                    tuple_map[c] = "hydro_ror_actual_entsoe"
                elif "Hydro Water Reservoir" in c[0]:
                    tuple_map[c] = "hydro_reservoir_actual_entsoe"
                elif "Hydro Pumped Storage" in c[0] or "Other storage" in c[0]:
                    tuple_map[c] = "hydro_pumped_actual_entsoe"
        if tuple_map:
            df = df.rename(columns=tuple_map)
    else:
        idx = pd.date_range(start=start, end=end, freq="1h", tz="UTC", inclusive="left")
        df = pd.DataFrame(index=idx)

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
    chunk_months: int = 1,
    workers: int = 1,
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
        merged = _apply_capacity_ffill(merged, cap_sparse)

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
    parser.add_argument("--chunk-months", type=int, default=1, help="Chunk size in months (default: 1).")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1).")
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
        merged = _apply_capacity_ffill(merged, cap_sparse)

    merged = merged.filter(pl.col("timestamp_utc") <= pl.lit(end))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(out_path, compression="zstd")
    LOGGER.info("Wrote %s rows to %s", len(merged), out_path)


if __name__ == "__main__":
    main()
