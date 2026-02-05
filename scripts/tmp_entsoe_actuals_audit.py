"""Audit ENTSO-E API request/response for Actual Generation vs Forecast.

Usage:
    ./.venv/bin/python scripts/tmp_entsoe_actuals_audit.py

Reads ENTSOE_API_KEY from env/.env and prints:
- Request parameters (domain, docType, processType)
- EIC codes
- TimeSeries counts and any domain codes found in response
"""
from __future__ import annotations

import os
from pathlib import Path
import xml.etree.ElementTree as ET
import requests
from dotenv import load_dotenv

# EIC codes
BIDDING_ZONE_DE_LU = "10Y1001A1001A83F"
CONTROL_AREAS = {
    "50Hertz": "10YDE-VE-------2",
    "TenneT": "10YDE-TENNET-GERM",
    "Amprion": "10YDE-AMPRION---8",
    "TransnetBW": "10YDE-ENBW-----N",
}

BASE_URL = "https://web-api.tp.entsoe.eu/api"


def load_env() -> None:
    for base in Path(__file__).resolve().parents:
        p = base / ".env"
        if p.exists():
            load_dotenv(p)
            return


def fetch_raw(params: dict) -> str:
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.text


def parse_timeseries(xml_text: str) -> dict:
    ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0"}
    root = ET.fromstring(xml_text)
    ts_nodes = root.findall(".//ns:TimeSeries", ns)
    domains = []
    for ts in ts_nodes:
        in_dom = ts.find(".//ns:in_Domain.mRID", ns)
        out_dom = ts.find(".//ns:out_Domain.mRID", ns)
        if in_dom is not None:
            domains.append(("in", in_dom.text))
        if out_dom is not None:
            domains.append(("out", out_dom.text))
    return {
        "timeseries_count": len(ts_nodes),
        "domains": domains,
    }


def main() -> None:
    load_env()
    key = os.getenv("ENTSOE_API_KEY")
    if not key:
        raise RuntimeError("Missing ENTSOE_API_KEY in env")

    # Example window
    period_start = "202401010000"
    period_end = "202401020000"

    # Base params
    base_actual = {
        "securityToken": key,
        "documentType": "A75",
        "processType": "A16",
        "periodStart": period_start,
        "periodEnd": period_end,
    }
    base_forecast = {
        "securityToken": key,
        "documentType": "A69",
        "processType": "A01",
        "periodStart": period_start,
        "periodEnd": period_end,
    }

    def _safe_params(p: dict) -> dict:
        out = dict(p)
        if "securityToken" in out:
            out["securityToken"] = "***"
        return out

    params_actual = {**base_actual, "in_Domain": BIDDING_ZONE_DE_LU, "out_Domain": BIDDING_ZONE_DE_LU}
    params_forecast = {**base_forecast, "in_Domain": BIDDING_ZONE_DE_LU, "out_Domain": BIDDING_ZONE_DE_LU}

    print("Actuals request params (DE-LU):", _safe_params(params_actual))
    print("Forecast request params (DE-LU):", _safe_params(params_forecast))
    print("Bidding zone EIC:", BIDDING_ZONE_DE_LU)
    print("Control area EICs:", CONTROL_AREAS)

    def _print_reason(xml_text: str, label: str) -> None:
        ns = {
            "ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0",
        }
        root = ET.fromstring(xml_text)
        reason = root.find(".//ns:Reason", ns)
        if reason is not None:
            text = reason.find(".//ns:text", ns)
            code = reason.find(".//ns:code", ns)
            print(f"{label} Reason:", (code.text if code is not None else None), (text.text if text is not None else None))

    xml_actual = fetch_raw(params_actual)
    xml_forecast = fetch_raw(params_forecast)

    meta_actual = parse_timeseries(xml_actual)
    meta_forecast = parse_timeseries(xml_forecast)

    print("Actuals TimeSeries count:", meta_actual["timeseries_count"])
    print("Actuals domains:", sorted(set(meta_actual["domains"]))[:20])
    print("Forecast TimeSeries count:", meta_forecast["timeseries_count"])
    print("Forecast domains:", sorted(set(meta_forecast["domains"]))[:20])

    if meta_actual["timeseries_count"] == 0:
        _print_reason(xml_actual, "Actuals")
    if meta_forecast["timeseries_count"] == 0:
        _print_reason(xml_forecast, "Forecast")

    # Try control-area domains for actuals to see if data exists there
    print("\n--- Control Area Actuals ---")
    for name, eic in CONTROL_AREAS.items():
        params = {**base_actual, "in_Domain": eic, "out_Domain": eic}
        xml = fetch_raw(params)
        meta = parse_timeseries(xml)
        print(f"{name} ({eic}) TimeSeries:", meta["timeseries_count"])
        if meta["timeseries_count"] == 0:
            _print_reason(xml, f"{name} Actuals")

    # Print first 500 chars for quick inspection
    print("\nActuals XML head:\n", xml_actual[:500])
    print("\nForecast XML head:\n", xml_forecast[:500])


if __name__ == "__main__":
    main()
