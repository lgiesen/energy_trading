"""Fetch ENTSO-E series used in the notebook and merge into one table.

Metrics produced (columns):
- system_load_forecast
- installed_generation_per_type
- activated_balancing_quantities_affr
- activated_balancing_quantities_mffr
- flow_net_* for DE-LU neighbors (AT, BE, CH, CZ, DK1, DK2, FR, NL, NO2, PL, SE4) and flow_net_total
- GEN_THERMAL, U_THERMAL, GEN_FOSSIL_BROWN_COAL_LIGNITE, GEN_FOSSIL_HARD_COAL, GEN_FOSSIL_GAS, GEN_NUCLEAR (thermal generation features)

The resulting DataFrame is aligned on timestamp and includes year/month/day.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import polars as pl
import requests
from dotenv import load_dotenv

from ingestion.entsoe_parser import combine_metric_responses


def _load_env():
    """Load .env by walking up the tree (also checking an energy_trading/ folder)."""
    for base in Path(__file__).resolve().parents:
        for candidate in (base / ".env", base / "energy_trading" / ".env"):
            if candidate.exists():
                load_dotenv(candidate)
                return candidate
    return None


def _build_urls(start: str, end: str, bidding_zone: str, process_year_ahead: str, api_key: str) -> Dict[str, str]:
    base = f"https://web-api.tp.entsoe.eu/api?periodStart={start}&periodEnd={end}&securityToken={api_key}"
    dt_system_total_load = "A65"
    dt_actual_generation_per_type = "A75"
    dt_installed_generation_per_type = "A68"
    dt_activated_balancing_quantities = "A83"
    pt_day_ahead = "A01"
    pt_realised = "A16"
    bt_afrr = "A96"
    bt_mfrr = "A97"
    dt_unavailability_generation = "A11"

    # Neighboring bidding zones for DE-LU.
    neighbor_bzn_eic = {
        "AT": "10YAT-APG------L",
        "BE": "10YBE----------2",
        "CH": "10YCH-SWISSGRIDZ",
        "CZ": "10YCZ-CEPS-----N",
        "DK1": "10YDK-1--------W",
        "DK2": "10YDK-2--------M",
        "FR": "10YFR-RTE------C",
        "NL": "10YNL----------L",
        "NO2": "10YNO-2--------T",
        "PL": "10YPL-AREA-----S",
        "SE4": "10Y1001A1001A47J",
    }

    urls = {
        "system_load_forecast": f"{base}&documentType={dt_system_total_load}&processType={pt_day_ahead}&outBiddingZone_Domain={bidding_zone}",
        "actual_generation_per_type": f"{base}&documentType={dt_actual_generation_per_type}&processType={pt_realised}&in_Domain={bidding_zone}",
        "installed_generation_per_type": f"{base}&documentType={dt_installed_generation_per_type}&processType={process_year_ahead}&in_Domain={bidding_zone}",
        "activated_balancing_quantities_affr": f"{base}&documentType={dt_activated_balancing_quantities}&businessType={bt_afrr}&controlArea_Domain={bidding_zone}",
        "activated_balancing_quantities_mffr": f"{base}&documentType={dt_activated_balancing_quantities}&businessType={bt_mfrr}&controlArea_Domain={bidding_zone}",
    }

    # Cross-border flows: import = into DE-LU, export = out of DE-LU.
    for cc, eic in neighbor_bzn_eic.items():
        # Observed direction: in_Domain appears to be the receiving side.
        urls[f"flow_import_{cc}"] = f"{base}&documentType={dt_unavailability_generation}&in_Domain={bidding_zone}&out_Domain={eic}"
        urls[f"flow_export_{cc}"] = f"{base}&documentType={dt_unavailability_generation}&in_Domain={eic}&out_Domain={bidding_zone}"

    return urls


def _make_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    """Create a requests session with simple retry on read timeouts."""
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


def _fetch(url: str, session: requests.Session, timeout: int = 60) -> str:
    resp = session.get(url, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        # Surface ENTSO-E error payload to make debugging easier.
        detail = resp.text[:500] if resp.text else "no response body"
        raise RuntimeError(f"ENTSO-E request failed ({resp.status_code}): {detail}") from exc
    return resp.text


def fetch_and_merge(start: str, end: str, bidding_zone: str, process_year_ahead: str = "A33", timeout: int = 60) -> pl.DataFrame:
    """Fetch all required series and merge on timestamp."""
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        _load_env()
        api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError("ENTSOE_API_KEY not set (load .env or export it).")

    session = _make_session()
    urls = _build_urls(start, end, bidding_zone, process_year_ahead, api_key)
    responses: Dict[str, str] = {}
    for metric, url in urls.items():
        try:
            responses[metric] = _fetch(url, session=session, timeout=timeout)
        except RuntimeError as exc:
            print(f"Warning: skipping metric '{metric}' due to fetch error: {exc}")
    merged = combine_metric_responses(responses)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Fetch ENTSO-E data and merge into one table.")
    parser.add_argument("--start", required=True, help="periodStart in UTC, e.g. 202201010000")
    parser.add_argument("--end", required=True, help="periodEnd in UTC, e.g. 202202010000")
    parser.add_argument("--bidding-zone", default="10Y1001A1001A82H", help="Bidding/Control area domain code (default DE-LU).")
    parser.add_argument("--process-year-ahead", default="A33", help="Process type for installed capacity (default A33).")
    parser.add_argument("--out", default="data/entsoe.parquet", help="Output parquet path.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP read timeout per request (seconds).")
    args = parser.parse_args()

    _load_env()
    df = fetch_and_merge(args.start, args.end, args.bidding_zone, args.process_year_ahead, timeout=args.timeout)
    if df.is_empty():
        print("No data parsed.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
