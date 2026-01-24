"""
Usage: 
    python -m ingestion.fetch_netztransparenz \
        --start 2022-01-01T00:00:00 --end 2025-12-31T23:45:00 \
        --out data/netztransparenz.parquet

Fetch NRV-Saldo, reBAP, and aFRR activation from Netztransparenz and merge on timestamp."""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv

TOTAL_AFRR_CAPACITY_MW = 2000  # rough national capacity used for activation rate

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
    # Clean BOM and line breaks to avoid parser issues.
    cleaned = text.lstrip("\ufeff").lstrip("ï»¿").replace("\r\n", "\n").replace("\r", "\n")
    return pl.read_csv(
        io.BytesIO(cleaned.encode("utf-8")),
        separator=";",
        infer_schema_length=2000,
        quote_char=None,
    )


def _normalize_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Trim and lower column names to make parsing resilient."""
    mapping = {c: c.strip() for c in df.columns}
    df = df.rename(mapping)
    mapping_lower = {c: c.lower() for c in df.columns}
    df = df.rename(mapping_lower)
    return df


def _tidy_nrv(df: pl.DataFrame) -> pl.DataFrame:
    df = _normalize_cols(df)
    # Use the last column as value if no explicit match.
    value_col = df.columns[-1]
    date_col = "datum" if "datum" in df.columns else df.columns[0]
    time_col = "von" if "von" in df.columns else df.columns[1]
    ts = (
        pl.concat_str([pl.col(date_col), pl.lit(" "), pl.col(time_col)])
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
        .select(
            [
                "timestamp_utc",
                pl.col(date_col).alias("datum"),
                pl.col("zeitzone") if "zeitzone" in df.columns else pl.lit(None).alias("zeitzone"),
                pl.col(time_col).alias("time"),
                "NRV_balance",
            ]
        )
    )


def _tidy_rebap(df: pl.DataFrame) -> pl.DataFrame:
    df = _normalize_cols(df)
    unter_col = next((c for c in df.columns if "unter" in c.lower()), df.columns[-2])
    ueber_col = next((c for c in df.columns if "ueber" in c.lower()), df.columns[-1])
    date_col = "datum" if "datum" in df.columns else df.columns[0]
    time_col = "von" if "von" in df.columns else df.columns[1]
    ts = (
        pl.concat_str([pl.col(date_col), pl.lit(" "), pl.col(time_col)])
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


def _tidy_afrr_activation(df: pl.DataFrame) -> pl.DataFrame:
    """Parse Aktivierte SRL (aFRR) volumes and derive activation rates."""
    df = _normalize_cols(df)
    date_col = "datum" if "datum" in df.columns else df.columns[0]
    time_col = "von" if "von" in df.columns else df.columns[1]
    ts = (
        pl.concat_str([pl.col(date_col), pl.lit(" "), pl.col(time_col)])
        .str.to_datetime("%d.%m.%Y %H:%M", strict=False)
        .dt.replace_time_zone("UTC")
        .alias("timestamp_utc")
    )
    pos_col = next((c for c in df.columns if "positiv" in c.lower()), None)
    neg_col = next((c for c in df.columns if "negativ" in c.lower()), None)
    if not pos_col or not neg_col:
        raise RuntimeError("Could not find positive/negative columns in Aktivierte SRL payload.")
    def _clean_num(col: str) -> pl.Expr:
        return (
            pl.col(col)
            .cast(pl.Utf8)
            .str.replace(".", "")
            .str.replace(",", ".")
            .cast(pl.Float64, strict=False)
        )
    clean = (
        df.with_columns(
            [
                ts,
                _clean_num(pos_col).alias("activated_volume_pos_mw"),
                _clean_num(neg_col).alias("activated_volume_neg_mw"),
            ]
        )
        .select(["timestamp_utc", "activated_volume_pos_mw", "activated_volume_neg_mw"])
        .with_columns(
            [
                (pl.col("activated_volume_pos_mw") / TOTAL_AFRR_CAPACITY_MW).alias("activation_rate_pos"),
                (pl.col("activated_volume_neg_mw") / TOTAL_AFRR_CAPACITY_MW).alias("activation_rate_neg"),
            ]
        )
    )
    return clean


def _chunk_range(start: datetime, end: datetime, days: int):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


def fetch_and_merge(start: str, end: str, token: str, timeout: int = 60, chunk_days: int = 180, chunk_sleep: float = 0.5) -> pl.DataFrame:
    base_url = "https://ds.netztransparenz.de/api/v1/data"
    session = _make_session(token)
    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))

    frames = []
    for s_dt, e_dt in _chunk_range(start_dt, end_dt, chunk_days):
        s_str = s_dt.isoformat(timespec="minutes")
        e_str = e_dt.isoformat(timespec="minutes")
        urls = {
            "nrv": f"{base_url}/NrvSaldo/NRVSaldo/Qualitaetsgesichert/{s_str}/{e_str}",
            "rebap": f"{base_url}/NrvSaldo/reBAP/Qualitaetsgesichert/{s_str}/{e_str}",
            "afrr": f"{base_url}/NrvSaldo/AktivierteSRL/Qualitaetsgesichert/{s_str}/{e_str}",
        }
        try:
            nrv_df = _tidy_nrv(_read_csv(_fetch_csv(urls["nrv"], session, timeout=timeout)))
            rebap_df = _tidy_rebap(_read_csv(_fetch_csv(urls["rebap"], session, timeout=timeout)))
            afrr_df = _tidy_afrr_activation(_read_csv(_fetch_csv(urls["afrr"], session, timeout=timeout)))
            merged = (
                nrv_df.join(rebap_df, on="timestamp_utc", how="full", coalesce=True)
                .join(afrr_df, on="timestamp_utc", how="full", coalesce=True)
                .sort("timestamp_utc")
            )
            frames.append(merged)
            print(f"Fetched chunk {s_str} -> {e_str}: {len(merged)} rows")
        except Exception as exc:
            print(f"Warning: chunk {s_str} -> {e_str} failed: {exc}")
        if chunk_sleep and e_dt < end_dt:
            time.sleep(chunk_sleep)

    if not frames:
        return pl.DataFrame()

    merged = pl.concat(frames).unique(subset=["timestamp_utc"], keep="last").sort("timestamp_utc")

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
    # Alias for imbalance
    if "NRV_balance" in merged.columns and "nrv_imbalance" not in merged.columns:
        merged = merged.with_columns(pl.col("NRV_balance").alias("nrv_imbalance"))

    return merged


def main():
    parser = argparse.ArgumentParser(description="Fetch NRV-Saldo and reBAP from Netztransparenz and merge on timestamp.")
    parser.add_argument("--start", required=True, help="Start ISO8601, e.g. 2022-01-01T00:00:00")
    parser.add_argument("--end", required=True, help="End ISO8601, e.g. 2022-12-31T23:45:00")
    parser.add_argument("--token", help="Bearer token (defaults to NETZTRANSPARENZ_TOKEN env var).")
    parser.add_argument("--out", default="data/netztransparenz.parquet", help="Output parquet path.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds per request.")
    parser.add_argument("--chunk-days", type=int, default=180, help="Chunk size in days to avoid server errors.")
    parser.add_argument("--chunk-sleep", type=float, default=0.5, help="Seconds to sleep between chunks.")
    args = parser.parse_args()

    _load_env()
    token = args.token or os.getenv("NETZTRANSPARENZ_TOKEN")
    if not token:
        client_id = os.getenv("NETZTRANSPARENZ_CLIENT_ID")
        client_secret = os.getenv("NETZTRANSPARENZ_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError("No token provided. Set NETZTRANSPARENZ_TOKEN or NETZTRANSPARENZ_CLIENT_ID/NETZTRANSPARENZ_CLIENT_SECRET.")
        token = _fetch_token_from_client_credentials(client_id, client_secret)

    df = fetch_and_merge(args.start, args.end, token, timeout=args.timeout, chunk_days=args.chunk_days, chunk_sleep=args.chunk_sleep)
    if df.is_empty():
        print("No data fetched.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
