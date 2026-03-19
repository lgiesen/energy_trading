#!/usr/bin/env python3
"""PoC: fetch ENTSO-E aFRR CBMP for one day and inspect time resolution.

Usage:
    ./.venv/bin/python scripts/poc_fetch_entsoe_cbmp.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient


def _print_resolution(df: pd.DataFrame) -> None:
    """Print inferred and empirical timestamp resolution."""
    if df.empty:
        print("No rows returned. Resolution cannot be inferred.")
        return

    idx = pd.DatetimeIndex(df.index).sort_values()
    inferred = idx.inferred_freq
    print(f"inferred_freq: {inferred}")

    if len(idx) < 2:
        print("Only one row returned. Not enough points for time delta check.")
        return

    deltas = idx.to_series().diff().dropna()
    if deltas.empty:
        print("Could not compute time deltas.")
        return

    first_delta = deltas.iloc[0]
    mode_delta = deltas.mode().iloc[0] if not deltas.mode().empty else first_delta
    print(f"delta_first_rows: {first_delta}")
    print(f"delta_mode: {mode_delta}")


def main() -> int:
    load_dotenv()
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        print("ERROR: ENTSOE_API_KEY is not set.")
        return 1

    client = EntsoePandasClient(api_key=api_key)

    # Exactly one day in strict UTC.
    start = pd.Timestamp("2025-02-26T00:00:00Z")
    end = pd.Timestamp("2025-02-27T00:00:00Z")
    area_candidates = [
        "DE_LU",              # aggregated bidding zone (often empty for CBMP)
        "10YDE-EON------1",   # TenneT GER (SCA)
        "10YDE-RWENET---I",   # Amprion (SCA)
        "10YDE-VE-------2",   # 50Hertz (SCA)
        "10YDE-ENBW-----N",   # TransnetBW (SCA)
    ]

    cbmp_kwargs = {
        "process_type": "A67",
        "business_type": "A96",
    }

    print("Request parameters:")
    print(f"  area_candidates: {area_candidates}")
    print(f"  start_utc:    {start}")
    print(f"  end_utc:      {end}")
    print(f"  kwargs:       {cbmp_kwargs}")

    df: pd.DataFrame | pd.Series | None = None
    used_area: str | None = None
    errors: list[tuple[str, str, str]] = []
    for area in area_candidates:
        try:
            print(f"\nTrying area: {area}")
            out = client.query_activated_balancing_energy_prices(
                country_code=area,
                start=start,
                end=end,
                **cbmp_kwargs,
            )
            df = out
            used_area = area
            print(f"Success with area: {area}")
            break
        except Exception as exc:
            errors.append((area, type(exc).__name__, str(exc)))
            print(f"Failed for {area}: {type(exc).__name__}: {exc}")

    if df is None:
        print("\nCBMP request failed for all area candidates.")
        for area, exc_type, msg in errors:
            print(f"  - {area}: {exc_type}: {msg}")
        return 2

    if isinstance(df, pd.Series):
        df = df.to_frame(name="value")

    print("\nResult preview:")
    print(f"used_area: {used_area}")
    print(df.head(10))
    print(f"\nrows: {len(df)}")
    print(f"columns: {list(df.columns)}")
    print(f"index_tz: {getattr(df.index, 'tz', None)}")

    print("\nResolution check:")
    _print_resolution(df)

    return 0


if __name__ == "__main__":
    sys.exit(main())
