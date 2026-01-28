"""Debug ENTSO-E outage events for a short window (raw inspection only).

Usage:
    python -m ingestion.debug_outages
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient

LOGGER = logging.getLogger(__name__)
DEFAULT_COUNTRY = "DE_LU"


def _load_env_fallback() -> None:
    if os.getenv("ENTSOE_API_KEY"):
        return
    candidates = [
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().parents[2] / ".env",
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _load_env_fallback()
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError("ENTSOE_API_KEY is not set.")

    client = EntsoePandasClient(api_key=api_key)
    start = pd.Timestamp(datetime(2023, 1, 1), tz="UTC")
    end = pd.Timestamp(datetime(2023, 1, 10), tz="UTC")

    df = client.query_unavailability_of_generation_units(
        DEFAULT_COUNTRY, start=start, end=end, docstatus="A05"
    )

    print("--- ENTSO-E Outage Debug ---")
    print(f"Rows returned: {len(df)}")
    if len(df) == 0:
        print("API returned NO data for this period/config.")
        return

    print("Columns:")
    print(df.columns.tolist())

    if "plant_type" in df.columns:
        print("\\nplant_type unique values:")
        print(df["plant_type"].dropna().unique())
        print("\\nplant_type value counts:")
        print(df["plant_type"].value_counts().head(20))
    elif "production_resource_psr_name" in df.columns:
        print("\\nproduction_resource_psr_name unique values:")
        print(df["production_resource_psr_name"].dropna().unique())
        print("\\nproduction_resource_psr_name value counts:")
        print(df["production_resource_psr_name"].value_counts().head(20))
    else:
        print("\\nNo plant_type/production_resource_psr_name column found.")

    print("\\nSample rows:")
    print(df.head(5))


if __name__ == "__main__":
    main()
