from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping

import polars as pl

# Common PTxxM/PTxxH resolutions in ENTSO-E transparency documents.
RESOLUTION_SECONDS = {
    "PT15M": 900,
    "PT30M": 1800,
    "PT60M": 3600,
}

THERMAL_PSR_TYPES: tuple[str, ...] = ("B02", "B05", "B04", "B14")
THERMAL_PSR_LABELS: dict[str, str] = {
    "B02": "GEN_FOSSIL_BROWN_COAL_LIGNITE",
    "B05": "GEN_FOSSIL_HARD_COAL",
    "B04": "GEN_FOSSIL_GAS",
    "B14": "GEN_NUCLEAR",
}
GEN_THERMAL_COL = "GEN_THERMAL"
U_THERMAL_COL = "U_THERMAL"


def _parse_iso_utc(value: str) -> datetime:
    """Parse ENTSO-E timestamps like 2022-01-01T00:00Z into aware datetimes."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _ts_for_position(start: datetime, resolution_seconds: int, position: int) -> datetime:
    return start + timedelta(seconds=resolution_seconds * (position - 1))


def parse_entsoe_timeseries(xml_content: str | bytes, *, metric_name: str) -> pl.DataFrame:
    """
    Stream-parse a single ENTSO-E XML response into a tidy Polars DataFrame.

    Parameters
    ----------
    xml_content: str | bytes
        Raw XML content (string or bytes).
    metric_name: str
        Logical metric name to attach to the series (e.g. "system_load_forecast").

    Returns
    -------
    Polars DataFrame with columns:
        - timestamp (UTC)
        - value (float)
        - metric (metric_name)
        - year, month, day (integers)
    """
    if isinstance(xml_content, str):
        buffer = io.BytesIO(xml_content.encode("utf-8"))
    else:
        buffer = io.BytesIO(xml_content)

    points: list[tuple[datetime, float]] = []
    period_start: datetime | None = None
    resolution_seconds: int | None = None

    # Stream through the document to keep memory usage low for long periods.
    for event, elem in ET.iterparse(buffer, events=("start", "end")):
        tag = elem.tag.rsplit("}", 1)[-1]  # strip namespace if present

        if event == "end" and tag == "timeInterval" and period_start is None:
            start_text = elem.findtext("./{*}start")
            if start_text:
                period_start = _parse_iso_utc(start_text)

        elif event == "end" and tag == "resolution" and resolution_seconds is None:
            res_text = (elem.text or "").strip()
            resolution_seconds = RESOLUTION_SECONDS.get(res_text)

        elif event == "end" and tag == "Point":
            if period_start is None or resolution_seconds is None:
                elem.clear()
                continue

            pos_text = elem.findtext("./{*}position")
            qty_text = elem.findtext("./{*}quantity")
            if pos_text is None or qty_text is None:
                elem.clear()
                continue

            try:
                pos = int(pos_text)
                qty = float(qty_text)
            except ValueError:
                elem.clear()
                continue

            ts = _ts_for_position(period_start, resolution_seconds, pos)
            points.append((ts, qty))

        # Free memory as we finish a Period subtree.
        if event == "end" and tag == "Period":
            elem.clear()

    if not points:
        return pl.DataFrame()

    df = pl.DataFrame(points, schema=["timestamp", "value"], orient="row").with_columns(
        [
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.month().alias("month"),
            pl.col("timestamp").dt.day().alias("day"),
            pl.lit(metric_name).alias("metric"),
        ]
    )
    return df


def _parse_generation_per_type(xml_content: str | bytes, allowed_psr_types: Iterable[str]) -> pl.DataFrame:
    """
    Parse "Actual Generation per Production Type" into hourly generation per psrType.

    Returns a DataFrame with columns: timestamp (UTC), psr_type, value.
    """
    if isinstance(xml_content, str):
        buffer = io.BytesIO(xml_content.encode("utf-8"))
    else:
        buffer = io.BytesIO(xml_content)

    allowed = set(allowed_psr_types)
    records: list[tuple[datetime, str, float]] = []
    current_psr: str | None = None
    period_start: datetime | None = None
    resolution_seconds: int | None = None

    for event, elem in ET.iterparse(buffer, events=("start", "end")):
        tag = elem.tag.rsplit("}", 1)[-1]

        if event == "start" and tag == "TimeSeries":
            current_psr = None
            period_start = None
            resolution_seconds = None

        elif event == "end" and tag == "psrType":
            current_psr = (elem.text or "").strip()

        elif event == "end" and tag == "timeInterval":
            start_text = elem.findtext("./{*}start")
            if start_text:
                period_start = _parse_iso_utc(start_text)

        elif event == "end" and tag == "resolution":
            res_text = (elem.text or "").strip()
            resolution_seconds = RESOLUTION_SECONDS.get(res_text)

        elif event == "end" and tag == "Point":
            if (
                current_psr not in allowed
                or period_start is None
                or resolution_seconds is None
            ):
                elem.clear()
                continue

            pos_text = elem.findtext("./{*}position")
            qty_text = elem.findtext("./{*}quantity")
            if pos_text is None or qty_text is None:
                elem.clear()
                continue

            try:
                pos = int(pos_text)
                qty = float(qty_text)
            except ValueError:
                elem.clear()
                continue

            ts = _ts_for_position(period_start, resolution_seconds, pos)
            records.append((ts, current_psr, qty))

        if event == "end" and tag == "Period":
            elem.clear()

        if event == "end" and tag == "TimeSeries":
            elem.clear()

    if not records:
        return pl.DataFrame()

    df = pl.DataFrame(records, schema=["timestamp", "psr_type", "value"], orient="row")
    hourly = (
        df.with_columns(pl.col("timestamp").dt.truncate("1h").alias("timestamp"))
        .lazy()
        .group_by(["timestamp", "psr_type"])
        .agg(pl.col("value").sum().alias("value"))
        .collect(streaming=False)
    )
    return hourly


def _thermal_features_from_generation(generation: pl.DataFrame) -> pl.DataFrame:
    """
    Create thermal generation features from a psr_type/value table.

    Output columns: timestamp, GEN_THERMAL, U_THERMAL, GEN_FOSSIL_BROWN_COAL_LIGNITE,
    GEN_FOSSIL_HARD_COAL, GEN_FOSSIL_GAS, GEN_NUCLEAR.
    """
    if generation.is_empty():
        return generation

    pivoted = (
        generation.pivot(index="timestamp", columns="psr_type", values="value", aggregate_function="first")
        .sort("timestamp")
    )

    for psr in THERMAL_PSR_TYPES:
        if psr not in pivoted.columns:
            pivoted = pivoted.with_columns(pl.lit(0.0).alias(psr))

    pivoted = pivoted.select(
        ["timestamp", *THERMAL_PSR_TYPES]
    ).with_columns(
        [pl.col(psr).fill_null(0.0).alias(psr) for psr in THERMAL_PSR_TYPES]
    )

    renamed = pivoted.rename({psr: THERMAL_PSR_LABELS[psr] for psr in THERMAL_PSR_TYPES})
    gen_cols = list(THERMAL_PSR_LABELS.values())
    thermal = renamed.with_columns(
        pl.sum_horizontal(gen_cols).alias(GEN_THERMAL_COL)
    )

    capacity = thermal.select(pl.col(GEN_THERMAL_COL).quantile(0.99)).item()
    capacity = float(capacity) if capacity is not None else 0.0

    thermal = thermal.with_columns(
        pl.when(capacity > 0.0)
        .then(pl.col(GEN_THERMAL_COL) / capacity)
        .otherwise(0.0)
        .alias(U_THERMAL_COL)
    )

    ordered_cols = ["timestamp", GEN_THERMAL_COL, U_THERMAL_COL, *gen_cols]
    return thermal.select(ordered_cols)


def combine_metric_responses(responses: Mapping[str, str | bytes]) -> pl.DataFrame:
    """
    Parse multiple XML responses and return a wide table with one column per metric.

    Parameters
    ----------
    responses: Mapping[str, str | bytes]
        Mapping of metric_name -> xml_content. Example keys:
        "system_load_forecast", "actual_generation_per_type",
        "installed_generation_per_type", "activated_balancing_quantities_affr",
        "activated_balancing_quantities_mffr".

    Returns
    -------
    Polars DataFrame with columns: timestamp, year, month, day, and one column per metric.
    """
    frames = []
    thermal_features = None
    for metric, xml_content in responses.items():
        if metric == "actual_generation_per_type":
            psr_df = _parse_generation_per_type(xml_content, allowed_psr_types=THERMAL_PSR_TYPES)
            if not psr_df.is_empty():
                thermal_features = _thermal_features_from_generation(psr_df).with_columns(
                    pl.col("timestamp") + pl.duration(hours=1)
                )
            continue

        df = parse_entsoe_timeseries(xml_content, metric_name=metric)
        if not df.is_empty():
            frames.append(df.with_columns(pl.col("timestamp") + pl.duration(hours=1)))

    base = pl.DataFrame()
    if frames:
        long_df = pl.concat(frames)
        base = (
            long_df.pivot(
                index="timestamp",
                columns="metric",
                values="value",
                aggregate_function="first",  # keep first if duplicates per metric/timestamp
            )
            .sort("timestamp")
        )

    if thermal_features is not None and not thermal_features.is_empty():
        if base.is_empty():
            base = thermal_features
        else:
            base = base.join(thermal_features, on="timestamp", how="outer")

    if base.is_empty():
        return base

    base = base.sort("timestamp").with_columns(
        [
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.month().alias("month"),
            pl.col("timestamp").dt.day().alias("day"),
        ]
    )

    metric_cols = [name for name in responses.keys() if name != "actual_generation_per_type" and name in base.columns]
    metric_cols += [col for col in (GEN_THERMAL_COL, U_THERMAL_COL, *THERMAL_PSR_LABELS.values()) if col in base.columns]
    ordered_cols = ["timestamp", "year", "month", "day", *metric_cols]
    return base.select(ordered_cols)
