"""Discover SMARD filter IDs for a given region/resolution.

Usage:
    ./.venv/bin/python -m energy_trading.utils.discover_smard_filters \
        --region DE-LU --resolution quarterhour \
        --start-id 1 --end-id 20000

    ./.venv/bin/python -m energy_trading.utils.discover_smard_filters \
        --search Steinkohle --resolution quarterhour

    ./.venv/bin/python -m energy_trading.utils.discover_smard_filters \
        --region DE-LU --resolution quarterhour \
        --ids 4169,715,716,717

Outputs:
    - Prints IDs that return a valid index_{resolution}.json with timestamps.
"""
from __future__ import annotations

import argparse
import time
from typing import Iterable, List, Optional

import requests

BASE_URL = "https://www.smard.de/app/chart_data"
MARKET_CONFIG_URL = "https://www.smard.de/app/chart_configuration/market_data_configuration.json"


def _iter_ids(args) -> Iterable[int]:
    if args.ids:
        return [int(x) for x in args.ids.split(",") if x.strip()]
    if args.start_id is None or args.end_id is None:
        raise ValueError("Provide --ids or --start-id/--end-id.")
    return range(args.start_id, args.end_id + 1)


def _check_id(filter_id: int, region: str, resolution: str) -> int:
    url = f"{BASE_URL}/{filter_id}/{region}/index_{resolution}.json"
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200:
        return 0
    try:
        payload = resp.json()
    except Exception:
        return 0
    return len(payload.get("timestamps", []))


def _fetch_market_config() -> dict:
    resp = requests.get(MARKET_CONFIG_URL, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _extract_modules(cfg: dict) -> List[dict]:
    modules: List[dict] = []

    def walk(obj) -> None:
        if isinstance(obj, dict):
            if "data_id" in obj and "name" in obj:
                modules.append(obj)
            for val in obj.values():
                walk(val)
        elif isinstance(obj, list):
            for val in obj:
                walk(val)

    walk(cfg)
    return modules


def _search_modules(modules: List[dict], query: str, resolution: Optional[str]) -> List[dict]:
    q = query.lower()
    hits = [m for m in modules if q in str(m.get("name", "")).lower()]
    if resolution:
        hits = [m for m in hits if m.get("source_resolution") == resolution]
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover SMARD filter IDs for a given region/resolution.")
    parser.add_argument("--region", default="DE-LU", help="SMARD region code (default DE-LU).")
    parser.add_argument("--resolution", default="quarterhour", help="Resolution (hour, quarterhour, day).")
    parser.add_argument(
        "--search",
        help="Search market_data_configuration.json for module names (e.g. Steinkohle).",
    )
    parser.add_argument("--ids", help="Comma-separated list of filter IDs to probe.")
    parser.add_argument("--start-id", type=int, help="Start ID (inclusive).")
    parser.add_argument("--end-id", type=int, help="End ID (inclusive).")
    parser.add_argument("--sleep", type=float, default=0.05, help="Sleep seconds between requests.")
    args = parser.parse_args()

    if args.search:
        cfg = _fetch_market_config()
        modules = _extract_modules(cfg)
        hits = _search_modules(modules, args.search, args.resolution)
        if not hits:
            print(f"No matches found for '{args.search}'.")
            return
        for m in hits:
            data_id = m.get("data_id")
            name = m.get("name")
            src_res = m.get("source_resolution")
            module_type = m.get("module_type")
            regions = ",".join(m.get("region", []) or [])
            print(f"{data_id}: {name} | resolution={src_res} | module_type={module_type} | region={regions}")
        return

    hits: List[int] = []
    for fid in _iter_ids(args):
        try:
            n = _check_id(fid, args.region, args.resolution)
        except Exception:
            n = 0
        if n > 0:
            hits.append(fid)
            print(f"{fid}: {n} timestamps")
        if args.sleep:
            time.sleep(args.sleep)

    if not hits:
        print("No valid filter IDs found in the provided range/list.")


if __name__ == "__main__":
    main()
