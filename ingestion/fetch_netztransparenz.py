"""Fetch NRV-Saldo and reBAP data from Netztransparenz and merge on timestamp."""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv


def _load_env():
    """Load .env by walking up the tree (also checking an energy_trading/ folder)."""
    for base in Path(__file__).resolve().parents:
        for candidate in (base / ".env", base / "energy_trading" / ".env"):
            if candidate.exists():
                load_dotenv(candidate)
                return candidate
    return None


def _ensure_bearer(token: str) -> str:
    """Prefix token with 'Bearer ' if not already provided."""
    prefix = "Bearer "
    return token if token.startswith(prefix) else f"{prefix}{token}"


def _fetch_token_from_client_credentials(client_id: str, client_secret: str) -> str:
    url = "https://identity.netztransparenz.de/users/connect/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(url, data=payload, headers=headers, timeout=60)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        detail = resp.text[:500] if resp.text else "no response body"
        raise RuntimeError(f"Token request failed ({resp.status_code}): {detail}") from exc

    data = resp.json()
    access_token = data.get("access_token")
    if not access_token:
        raise RuntimeError("Token response missing access_token.")
    return _ensure_bearer(access_token)


def _make_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"Authorization": _ensure_bearer(token)})
    return s


def _fetch_csv(url: str, session: requests.Session, timeout: int = 60) -> str:
    resp = session.get(url, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        detail = resp.text[:500] if resp.text else "no response body"
        raise RuntimeError(f"Netztransparenz request failed ({resp.status_code}): {detail}") from exc
    return resp.text


def _read_csv(text: str) -> pl.DataFrame:
    return pl.read_csv(
        io.BytesIO(text.encode("utf-8")),
        separator=";",
        infer_schema_length=2000,
        quote_char=None,
    )


def _tidy_nrv(df: pl.DataFrame) -> pl.DataFrame:
    # Use the last column as value if no explicit match.
    value_col = df.columns[-1]
    ts = (
        pl.concat_str([pl.col("Datum"), pl.lit(" "), pl.col("von")])
        .str.to_datetime("%d.%m.%Y %H:%M", strict=False)
        .dt.replace_time_zone("UTC")
        .alias("timestamp_utc")
    )
    return (
        df.with_columns(
            [
                ts,
                pl.when(pl.col(value_col) == "N.A.")
                .then(None)
                .otherwise(pl.col(value_col))
                .cast(pl.Utf8)
                .str.replace(",", ".")
                .cast(pl.Float64, strict=False)
                .alias("NRV_balance"),
            ]
        )
        .select(["timestamp_utc", "Datum", "Zeitzone", pl.col("von").alias("time"), "NRV_balance"])
    )


def _tidy_rebap(df: pl.DataFrame) -> pl.DataFrame:
    unter_col = next((c for c in df.columns if "unter" in c.lower()), df.columns[-2])
    ueber_col = next((c for c in df.columns if "ueber" in c.lower()), df.columns[-1])
    ts = (
        pl.concat_str([pl.col("Datum"), pl.lit(" "), pl.col("von")])
        .str.to_datetime("%d.%m.%Y %H:%M", strict=False)
        .dt.replace_time_zone("UTC")
        .alias("timestamp_utc")
    )
    return (
        df.with_columns(
            [
                ts,
                pl.when(pl.col(unter_col) == "N.A.")
                .then(None)
                .otherwise(pl.col(unter_col))
                .cast(pl.Utf8)
                .str.replace(",", ".")
                .cast(pl.Float64, strict=False)
                .alias("reBAP unterdeckt"),
                pl.when(pl.col(ueber_col) == "N.A.")
                .then(None)
                .otherwise(pl.col(ueber_col))
                .cast(pl.Utf8)
                .str.replace(",", ".")
                .cast(pl.Float64, strict=False)
                .alias("reBAP ueberdeckt"),
            ]
        )
        .select(
            ["timestamp_utc", "Datum", "Zeitzone", pl.col("von").alias("time"), "reBAP unterdeckt", "reBAP ueberdeckt"]
        )
    )


def fetch_and_merge(start: str, end: str, token: str, timeout: int = 60) -> pl.DataFrame:
    base_url = "https://ds.netztransparenz.de/api/v1/data"
    nrv_url = f"{base_url}/NrvSaldo/NRVSaldo/Qualitaetsgesichert/{start}/{end}"
    rebap_url = f"{base_url}/NrvSaldo/reBAP/Qualitaetsgesichert/{start}/{end}"

    session = _make_session(token)
    nrv_df = _tidy_nrv(_read_csv(_fetch_csv(nrv_url, session, timeout=timeout)))
    rebap_df = _tidy_rebap(_read_csv(_fetch_csv(rebap_url, session, timeout=timeout)))

    merged = nrv_df.join(rebap_df, on="timestamp_utc", how="full", coalesce=True).sort("timestamp_utc")

    # Drop duplicate metadata columns from the right-hand frame if they exist.
    drop_cols = [c for c in ("Datum_right", "Zeitzone_right", "von_right", "time_right") if c in merged.columns]
    if drop_cols:
        merged = merged.drop(drop_cols)

    # Drop TSO-specific columns if present (not needed for ML) and reduce to a single datetime column.
    tso_cols = [
        "50Hertz (Positiv)",
        "Amprion (Positiv)",
        "TenneT TSO (Positiv)",
        "TransnetBW (Positiv)",
        "50Hertz (Negativ)",
        "Amprion (Negativ)",
        "TenneT TSO (Negativ)",
        "TransnetBW (Negativ)",
        "50Hertz",
        "Amprion",
        "TenneT TSO",
        "TransnetBW",
    ]
    merged = merged.drop([c for c in tso_cols if c in merged.columns])

    # Keep only one datetime column for merging; drop redundant date/time fields if present.
    merged = merged.drop([c for c in ("Datum", "Zeitzone", "time") if c in merged.columns])

    return merged


def main():
    parser = argparse.ArgumentParser(description="Fetch NRV-Saldo and reBAP from Netztransparenz and merge on timestamp.")
    parser.add_argument("--start", required=True, help="Start ISO8601, e.g. 2022-01-01T00:00:00")
    parser.add_argument("--end", required=True, help="End ISO8601, e.g. 2022-12-31T23:45:00")
    parser.add_argument("--token", help="Bearer token (defaults to NETZTRANSPARENZ_TOKEN env var).")
    parser.add_argument("--out", default="data/netztransparenz.parquet", help="Output parquet path.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds per request.")
    args = parser.parse_args()

    _load_env()
    token = args.token or os.getenv("NETZTRANSPARENZ_TOKEN")
    if not token:
        client_id = os.getenv("NETZTRANSPARENZ_CLIENT_ID")
        client_secret = os.getenv("NETZTRANSPARENZ_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError("No token provided. Set NETZTRANSPARENZ_TOKEN or NETZTRANSPARENZ_CLIENT_ID/NETZTRANSPARENZ_CLIENT_SECRET.")
        token = _fetch_token_from_client_credentials(client_id, client_secret)

    df = fetch_and_merge(args.start, args.end, token, timeout=args.timeout)
    if df.is_empty():
        print("No data fetched.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
