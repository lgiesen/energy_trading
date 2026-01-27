"""Discover SMARD filter IDs for a given region/resolution.

Usage:
    python -m ingestion.discover_smard_filters \
        --region DE-LU --resolution quarterhour \
        --start-id 1 --end-id 20000

    python -m ingestion.discover_smard_filters \
        --region DE-LU --resolution quarterhour \
        --ids 4169,715,716,717

Outputs:
    - Prints IDs that return a valid index_{resolution}.json with timestamps.
"""
from __future__ import annotations

import argparse
import time
from typing import Iterable, List

import requests

BASE_URL = "https://www.smard.de/app/chart_data"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover SMARD filter IDs for a given region/resolution.")
    parser.add_argument("--region", default="DE-LU", help="SMARD region code (default DE-LU).")
    parser.add_argument("--resolution", default="quarterhour", help="Resolution (hour, quarterhour, day).")
    parser.add_argument("--ids", help="Comma-separated list of filter IDs to probe.")
    parser.add_argument("--start-id", type=int, help="Start ID (inclusive).")
    parser.add_argument("--end-id", type=int, help="End ID (inclusive).")
    parser.add_argument("--sleep", type=float, default=0.05, help="Sleep seconds between requests.")
    args = parser.parse_args()

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
