#!/usr/bin/env python3
"""Fetch planned and unplanned generation outages from ENTSO-E for DE-LU.

This script uses `entsoe-py` for authentication, request handling, and XML/ZIP
parsing. It queries ENTSO-E Transparency Platform generation unavailability
messages (`documentType=A80`) and filters by business type:

- Planned outages:   `businessType=A53`
- Unplanned outages: `businessType=A54`

Output columns:
- mrid
- unit_name
- nominal_power
- unavailable_power
- start
- end
- reason

Usage:
    ./.venv/bin/python scripts/fetch_entsoe_outages.py
    ./.venv/bin/python scripts/fetch_entsoe_outages.py --days-ahead 7
    ./.venv/bin/python scripts/fetch_entsoe_outages.py --start 2026-03-20T00:00:00Z --end 2026-03-27T00:00:00Z

Environment:
    ENTSOE_API_TOKEN or ENTSOE_API_KEY
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.mappings import Area, lookup_area
from entsoe.parsers import parse_unavailabilities

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


LOGGER = logging.getLogger(__name__)

# entsoe-py maps DE_LU to this EIC. The frequently seen `...82L` variant is not
# the code used by the library mappings and should not be used here.
DE_LU_BIDDING_ZONE_CODE = "10Y1001A1001A82H"
DOCUMENT_TYPE_GENERATION_UNAVAILABILITY = "A80"
BUSINESS_TYPE_PLANNED = "A53"
BUSINESS_TYPE_UNPLANNED = "A54"

BUSINESS_TYPE_REASON = {
    BUSINESS_TYPE_PLANNED: "Planned maintenance",
    BUSINESS_TYPE_UNPLANNED: "Unplanned outage",
}
DEFAULT_HISTORY_START_UTC = pd.Timestamp("2020-11-29T23:00:00Z")
DEFAULT_CHUNK_DAYS = 90
ENTSOE_PAGE_SIZE = 200


def _parse_utc(ts: str) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        out = out.tz_localize("UTC")
    else:
        out = out.tz_convert("UTC")
    return out


def _get_client() -> EntsoePandasClient:
    if load_dotenv is not None:
        load_dotenv(dotenv_path=Path(".env"))

    token = os.getenv("ENTSOE_API_TOKEN") or os.getenv("ENTSOE_API_KEY")
    if not token:
        raise RuntimeError("Missing ENTSOE_API_TOKEN (or ENTSOE_API_KEY) environment variable")
    return EntsoePandasClient(api_key=token)


def _query_generation_unavailability(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    business_type: str,
) -> pd.DataFrame:
    """Query ENTSO-E generation unavailability and parse to DataFrame.

    We use the entsoe-py low-level request path so we can pass `businessType`
    explicitly, then parse the ZIP/XML payload with the library parser.
    """
    area = lookup_area(Area.DE_LU)
    pages: list[pd.DataFrame] = []
    offset = 0
    while True:
        params = {
            "documentType": DOCUMENT_TYPE_GENERATION_UNAVAILABILITY,
            # Use the same canonical key casing as entsoe-py internals.
            "biddingZone_domain": area.code,
            "businessType": business_type,
            "offset": offset,
        }
        response = client._base_request(params=params, start=start, end=end)
        page = parse_unavailabilities(response.content, DOCUMENT_TYPE_GENERATION_UNAVAILABILITY)
        if page.empty:
            break

        # entsoe-py returns the parsed frame indexed by created_doc_time in area tz.
        if not isinstance(page.index, pd.DatetimeIndex):
            page.index = pd.to_datetime(page.index, utc=True, errors="coerce")
        elif page.index.tz is None:
            page.index = page.index.tz_localize(area.tz).tz_convert("UTC")
        else:
            page.index = page.index.tz_convert("UTC")

        for col in ("start", "end"):
            if col in page.columns:
                page[col] = pd.to_datetime(page[col], utc=True, errors="coerce")

        pages.append(page)
        if len(page) < ENTSOE_PAGE_SIZE:
            break
        offset += ENTSOE_PAGE_SIZE

    if not pages:
        return pd.DataFrame()
    return pd.concat(pages, axis=0, ignore_index=False)


def _query_generation_unavailability_chunked(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    business_type: str,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
) -> pd.DataFrame:
    """Fetch outage events in chunks to avoid oversized ENTSO-E requests."""
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    cur = start
    step = pd.Timedelta(days=max(1, int(chunk_days)))

    while cur < end:
        cur_end = min(cur + step, end)
        LOGGER.info("Fetching %s chunk [%s, %s)", business_type, cur, cur_end)
        try:
            chunk = _query_generation_unavailability(
                client=client,
                start=cur,
                end=cur_end,
                business_type=business_type,
            )
        except Exception as exc:  # pragma: no cover - network/API behavior
            msg = f"Chunk [{cur}, {cur_end}) failed: {exc}"
            LOGGER.error(msg)
            errors.append(msg)
            cur = cur_end
            continue
        if not chunk.empty:
            frames.append(chunk)
        cur = cur_end

    if not frames:
        if errors:
            raise RuntimeError(
                f"All chunk requests failed for businessType={business_type}. "
                f"First error: {errors[0]}"
            )
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=False)


def _clean_outages(df: pd.DataFrame, business_type: str) -> pd.DataFrame:
    """Normalize outage frame to the requested production schema."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "mrid",
                "unit_name",
                "nominal_power",
                "unavailable_power",
                "start",
                "end",
                "reason",
            ]
        )

    out = df.reset_index().rename(columns={"created_doc_time": "message_created_at"}).copy()

    out["nominal_power"] = pd.to_numeric(out.get("nominal_power"), errors="coerce")
    out["avail_qty"] = pd.to_numeric(out.get("avail_qty"), errors="coerce")
    out["unavailable_power"] = out["nominal_power"] - out["avail_qty"]

    # Guard against parser/source quirks that could produce negative unavailability.
    out.loc[out["unavailable_power"] < 0, "unavailable_power"] = pd.NA

    out["reason"] = out.get("businesstype")
    if "reason" in out.columns:
        out["reason"] = out["reason"].fillna(BUSINESS_TYPE_REASON[business_type])
    else:
        out["reason"] = BUSINESS_TYPE_REASON[business_type]

    cleaned = out.rename(
        columns={
            "production_resource_name": "unit_name",
        }
    )

    # Keep the latest version of a message for a given outage interval.
    dedupe_keys = [
        "mrid",
        "production_resource_id",
        "start",
        "end",
    ]
    dedupe_keys = [c for c in dedupe_keys if c in cleaned.columns]
    cleaned = cleaned.sort_values(["message_created_at", "revision"], na_position="last")
    if dedupe_keys:
        cleaned = cleaned.drop_duplicates(subset=dedupe_keys, keep="last")

    cleaned = cleaned[
        [
            c
            for c in [
                "mrid",
                "unit_name",
                "nominal_power",
                "unavailable_power",
                "start",
                "end",
                "reason",
            ]
            if c in cleaned.columns
        ]
    ].copy()

    cleaned = cleaned.sort_values(["start", "unit_name", "mrid"], na_position="last").reset_index(drop=True)
    return cleaned


def get_planned_outages(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch planned generation outages (A80 + A53) for DE-LU."""
    client = _get_client()
    try:
        raw = _query_generation_unavailability_chunked(
            client=client,
            start=start,
            end=end,
            business_type=BUSINESS_TYPE_PLANNED,
        )
    except Exception as exc:  # pragma: no cover - network/API behavior
        LOGGER.error("Failed to fetch planned outages: %s", exc)
        return _clean_outages(pd.DataFrame(), BUSINESS_TYPE_PLANNED)
    return _clean_outages(raw, BUSINESS_TYPE_PLANNED)


def get_unplanned_outages(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch unplanned generation outages (A80 + A54) for DE-LU."""
    client = _get_client()
    try:
        raw = _query_generation_unavailability_chunked(
            client=client,
            start=start,
            end=end,
            business_type=BUSINESS_TYPE_UNPLANNED,
        )
    except Exception as exc:  # pragma: no cover - network/API behavior
        LOGGER.error("Failed to fetch unplanned outages: %s", exc)
        return _clean_outages(pd.DataFrame(), BUSINESS_TYPE_UNPLANNED)
    return _clean_outages(raw, BUSINESS_TYPE_UNPLANNED)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fetch planned and unplanned ENTSO-E generation outages for DE-LU.")
    parser.add_argument("--start", help="UTC start timestamp, e.g. 2026-03-20T00:00:00Z")
    parser.add_argument("--end", help="UTC end timestamp, e.g. 2026-03-27T00:00:00Z")
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=7,
        help=(
            "If --end is omitted, extend end by N days from start. "
            "If both --start/--end are omitted, start defaults to project history start."
        ),
    )
    parser.add_argument("--out-dir", default="data/raw/entsoe_outages", help="Output directory for parquet/csv files.")
    args = parser.parse_args()

    now_utc = pd.Timestamp.now("UTC")
    start = _parse_utc(args.start) if args.start else DEFAULT_HISTORY_START_UTC
    end = _parse_utc(args.end) if args.end else (now_utc.ceil("h") + pd.Timedelta(days=args.days_ahead))

    if end <= start:
        raise ValueError("--end must be after --start")

    LOGGER.info("Fetching DE-LU planned outages from %s to %s", start, end)
    planned = get_planned_outages(start, end)
    LOGGER.info("Fetched %s planned outage rows", len(planned))

    LOGGER.info("Fetching DE-LU unplanned outages from %s to %s", start, end)
    unplanned = get_unplanned_outages(start, end)
    LOGGER.info("Fetched %s unplanned outage rows", len(unplanned))

    if planned.empty and unplanned.empty:
        raise RuntimeError(
            "Outage fetch returned 0 rows for both planned and unplanned outages. "
            "Check ENTSOE_API_TOKEN/ENTSOE_API_KEY and request range."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    planned_parquet = out_dir / "planned_generation_outages.parquet"
    unplanned_parquet = out_dir / "unplanned_generation_outages.parquet"
    planned_csv = out_dir / "planned_generation_outages.csv"
    unplanned_csv = out_dir / "unplanned_generation_outages.csv"

    planned.to_parquet(planned_parquet, index=False)
    unplanned.to_parquet(unplanned_parquet, index=False)
    planned.to_csv(planned_csv, index=False)
    unplanned.to_csv(unplanned_csv, index=False)

    print("\nPlanned outages:")
    print(planned.head(10).to_string(index=False))
    print("\nUnplanned outages:")
    print(unplanned.head(10).to_string(index=False))
    print(f"\nWrote: {planned_parquet}")
    print(f"Wrote: {unplanned_parquet}")


if __name__ == "__main__":
    main()
