"""
Usage:
    ./.venv/bin/python -m energy_trading.ingestion.fetch_netztransparenz \
        --start 2020-11-30T23:00:00Z \
        --end 2025-12-31T23:00:00Z \
        --chunk-days 30 --chunk-sleep 3 \
        --out data/raw/netztransparenz.parquet

Outputs:
    - netztransparenz.parquet with hourly data aligned on timestamp_utc (UTC).

Includes:
    - NRV-Saldo (NRV_balance)
    - reBAP (reBAP_shortage_surplus)
    - aFRR/mFRR activated volumes (afrr_activated_mw_*, mfrr_activated_mw_*)
    - RZ-Saldo (rz_saldo_mw + per-TSO columns)

Auth:
    - Reads credentials/token from .env (NETZTRANSPARENZ_* variables).
    - Token can be direct bearer token or client-credentials based.
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Tuple
from zoneinfo import ZoneInfo

import polars as pl
import requests
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)

# SMARD fallback for RZ saldo
SMARD_BASE_URL = "https://www.smard.de/app/chart_data"
SMARD_RZ_SALDO_ID = 37  # Regelzonensaldo (Netto)
SMARD_REGIONS = ["DE", "DE-LU"]
SMARD_RESOLUTION = "hour"


def _parse_mixed_numeric(value: object) -> float | None:
    """Parse mixed German/English numeric formats robustly.

    Handles values such as:
    - 1.105,35  -> 1105.35
    - -210,928  -> -210.928
    - -210.928  -> -210.928  (dot kept as decimal if comma absent)
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "n.a.", "n/a"}:
        return None

    s = re.sub(r"[^0-9,.-]", "", s)
    if not s:
        return None
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    # dot-only values are interpreted as decimal notation
    try:
        return float(s)
    except ValueError:
        return None


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
    s.headers.update({
        "Authorization": _ensure_bearer(token),
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        ),
    })
    return s


def _fetch_csv(url: str, session: requests.Session, timeout: int = 60, retries: int = 3, retry_sleep: float = 10.0) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                detail = resp.text[:500] if resp.text else "no response body"
                raise RuntimeError(f"Netztransparenz request failed ({resp.status_code}): {detail}") from exc
            return resp.text
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(retry_sleep)
            else:
                raise
    raise last_exc  # pragma: no cover


def _read_csv(text: str) -> pl.DataFrame:
    # Clean BOM and line breaks to avoid parser issues.
    cleaned = text.lstrip("\ufeff").lstrip("ï»¿").replace("\r\n", "\n").replace("\r", "\n")
    return pl.read_csv(
        io.BytesIO(cleaned.encode("utf-8")),
        separator=";",
        infer_schema_length=2000,
        quote_char=None,
    )


def _smard_available_timestamps(filter_id: int, region: str, resolution: str, session: requests.Session) -> List[int]:
    url = f"{SMARD_BASE_URL}/{filter_id}/{region}/index_{resolution}.json"
    resp = session.get(url, timeout=30)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    return sorted(resp.json().get("timestamps", []))


def _smard_fetch_chunk(filter_id: int, region: str, resolution: str, ts: int, session: requests.Session) -> List[Tuple[int, float]]:
    filename = f"{filter_id}_{region}_{resolution}_{ts}.json"
    url = f"{SMARD_BASE_URL}/{filter_id}/{region}/{filename}"
    resp = session.get(url, timeout=30)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    payload = resp.json()
    return [(int(k), float(v)) for k, v in payload.get("series", [])]


def _smard_series_to_frame(
    name: str,
    series_data: Iterable[Tuple[int, float]],
    start_ms: int,
    end_ms: int,
) -> pl.DataFrame:
    df = pl.DataFrame(series_data, schema=[("timestamp_ms", pl.Int64), (name, pl.Float64)], orient="row")
    df = (
        df.filter((pl.col("timestamp_ms") >= start_ms) & (pl.col("timestamp_ms") <= end_ms))
        .with_columns(pl.from_epoch("timestamp_ms", time_unit="ms").alias("timestamp_utc"))
        .with_columns(pl.col("timestamp_utc").dt.replace_time_zone("UTC"))
        .drop("timestamp_ms")
        .with_columns(pl.col("timestamp_utc").dt.truncate("1h").alias("timestamp_utc"))
        .group_by("timestamp_utc")
        .agg(pl.col(name).mean())
    )
    return df


def fetch_smard_saldo(start_utc: datetime, end_utc: datetime) -> pl.DataFrame:
    """Fetch RZ saldo from SMARD and return timestamp_utc + rz_saldo_mw."""
    session = requests.Session()
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)
    cutoff_ms = start_ms

    records: List[Tuple[int, float]] = []
    for region in SMARD_REGIONS:
        timestamps = _smard_available_timestamps(SMARD_RZ_SALDO_ID, region, SMARD_RESOLUTION, session)
        relevant = [ts for ts in timestamps if ts >= cutoff_ms]
        if not relevant:
            continue
        for ts in relevant:
            try:
                records.extend(_smard_fetch_chunk(SMARD_RZ_SALDO_ID, region, SMARD_RESOLUTION, ts, session))
            except requests.HTTPError as exc:
                LOGGER.warning("SMARD RZ saldo chunk %s failed (%s).", ts, exc)
        if records:
            break

    if not records:
        return pl.DataFrame()
    return _smard_series_to_frame("rz_saldo_mw", records, start_ms, end_ms)


def _normalize_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Trim and lower column names to make parsing resilient."""
    mapping = {c: c.strip() for c in df.columns}
    df = df.rename(mapping)
    mapping_lower = {c: c.lower() for c in df.columns}
    df = df.rename(mapping_lower)
    return df


def _find_region_col(df: pl.DataFrame) -> str | None:
    candidates = [
        "regelzone",
        "netzregelverbund",
        "gebiet",
        "region",
        "tso",
        "uebertragungsnetzbetreiber",
        "bilanz",
    ]
    for c in df.columns:
        for needle in candidates:
            if needle in c:
                return c
    return None


def _select_germany_rows(df: pl.DataFrame, region_col: str) -> tuple[pl.DataFrame, str | None]:
    # Prefer explicit Germany row only.
    # Do not select generic netting/IGCC rows here, as those can understate
    # activated aFRR/mFRR volumes compared with Germany totals.
    keywords = ["deutschland", "germany"]
    pattern = "|".join(keywords)
    try:
        matches = df.filter(pl.col(region_col).cast(pl.Utf8).str.contains(pattern, case=False, literal=False))
    except Exception:
        return df, None
    if matches.height == 0:
        return df, None
    label = matches.select(pl.col(region_col).cast(pl.Utf8)).unique().to_series().to_list()
    selected = label[0] if label else None
    return matches, selected


def _tidy_nrv(df: pl.DataFrame) -> pl.DataFrame:
    df = _normalize_cols(df)
    meta_cols = {
        "datum",
        "von",
        "time",
        "zeitzone",
        "bis",
        "einheit",
        "datenkategorie",
        "datentyp",
    }
    candidates = [c for c in df.columns if c not in meta_cols]
    # Exclude traffic-light/status style columns (e.g., "NRV-Saldo-Ampel").
    candidates = [c for c in candidates if "ampel" not in c]
    if not candidates:
        candidates = [c for c in df.columns if c not in meta_cols] or [df.columns[-1]]

    # Prefer explicit Germany value column for NRV saldo endpoints.
    value_col = next((c for c in candidates if "deutsch" in c), None)
    if value_col is None:
        value_col = next((c for c in candidates if ("nrv" in c and "saldo" in c)), None)
    if value_col is None:
        value_col = next((c for c in candidates if "saldo" in c), None)
    if value_col is None:
        value_col = candidates[-1]
    LOGGER.debug("NRV parser selected value column: %s", value_col)

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
                pl.when(pl.col(value_col).cast(pl.Utf8) == "N.A.")
                .then(None)
                .otherwise(pl.col(value_col))
                .cast(pl.Utf8)
                .map_elements(_parse_mixed_numeric, return_dtype=pl.Float64)
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
                pl.when(pl.col(unter_col).cast(pl.Utf8) == "N.A.")
                .then(None)
                .otherwise(pl.col(unter_col))
                .cast(pl.Utf8)
                .str.replace_all(",", ".", literal=True)
                .cast(pl.Float64, strict=False)
                .alias("reBAP unterdeckt"),
                pl.when(pl.col(ueber_col).cast(pl.Utf8) == "N.A.")
                .then(None)
                .otherwise(pl.col(ueber_col))
                .cast(pl.Utf8)
                .str.replace_all(",", ".", literal=True)
                .cast(pl.Float64, strict=False)
                .alias("reBAP ueberdeckt"),
            ]
        )
        .select(
            [
                "timestamp_utc",
                pl.col(date_col).alias("datum"),
                pl.col("zeitzone") if "zeitzone" in df.columns else pl.lit(None).alias("zeitzone"),
                pl.col(time_col).alias("time"),
                "reBAP unterdeckt",
                "reBAP ueberdeckt",
            ]
        )
        .with_columns(
            pl.coalesce(["reBAP unterdeckt", "reBAP ueberdeckt"]).alias("reBAP_shortage_surplus")
        )
        .drop(["reBAP unterdeckt", "reBAP ueberdeckt"])
    )


def _tidy_rz_saldo(df: pl.DataFrame) -> pl.DataFrame:
    """Parse RZ-Saldo (per TSO) and compute national total."""
    df = _normalize_cols(df)
    date_col = "datum" if "datum" in df.columns else df.columns[0]
    time_col = "von" if "von" in df.columns else df.columns[1]
    ts = (
        pl.concat_str([pl.col(date_col), pl.lit(" "), pl.col(time_col)])
        .str.to_datetime("%d.%m.%Y %H:%M", strict=False)
        .dt.replace_time_zone("UTC")
        .alias("timestamp_utc")
    )
    tso_cols = [c for c in df.columns if c in {"50hertz", "amprion", "tennet tso", "transnetbw"}]
    if not tso_cols:
        # Fall back: numeric columns after metadata
        tso_cols = df.columns[7:]

    def _clean_num(col: str) -> pl.Expr:
        return (
            pl.col(col)
            .cast(pl.Utf8)
            .str.replace_all(".", "", literal=True)
            .str.replace_all(",", ".", literal=True)
            .cast(pl.Float64, strict=False)
        )

    cleaned = df.with_columns([ts] + [_clean_num(c).alias(c) for c in tso_cols])
    total = None
    for c in tso_cols:
        total = pl.col(c) if total is None else (total + pl.col(c))
    return cleaned.select(["timestamp_utc"] + tso_cols).with_columns(total.alias("rz_saldo_mw"))


def _tidy_activation(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    df = _normalize_cols(df)
    date_col = "datum" if "datum" in df.columns else df.columns[0]
    time_col = "von" if "von" in df.columns else df.columns[1]
    region_col = _find_region_col(df)
    ts = (
        pl.concat_str([pl.col(date_col), pl.lit(" "), pl.col(time_col)])
        .str.to_datetime("%d.%m.%Y %H:%M", strict=False)
        .dt.replace_time_zone("UTC")
        .alias("timestamp_utc")
    )
    meta_cols = {date_col, time_col, "zeitzone"}
    if region_col:
        meta_cols = meta_cols | {region_col}
    pos_cols = [c for c in df.columns if "positiv" in c and c not in meta_cols]
    neg_cols = [c for c in df.columns if "negativ" in c and c not in meta_cols]
    if not pos_cols or not neg_cols:
        raise RuntimeError(f"Could not find positive/negative columns in {prefix} payload.")

    # Prefer Germany total columns when present to avoid summing duplicated
    # regional views in mixed payload layouts.
    pos_cols_de = [c for c in pos_cols if "deutschland" in c]
    neg_cols_de = [c for c in neg_cols if "deutschland" in c]
    use_de_cols = bool(pos_cols_de and neg_cols_de)
    if use_de_cols:
        pos_cols = pos_cols_de
        neg_cols = neg_cols_de

    def _clean_num(col: str) -> pl.Expr:
        return (
            pl.col(col)
            .cast(pl.Utf8)
            .str.replace_all(".", "", literal=True)
            .str.replace_all(",", ".", literal=True)
            .cast(pl.Float64, strict=False)
        )

    clean = df.with_columns([ts] + [_clean_num(c).alias(c) for c in (pos_cols + neg_cols)])
    selected_region = None
    if region_col and not use_de_cols:
        clean, selected_region = _select_germany_rows(clean, region_col)
    if selected_region:
        LOGGER.info("Selected region '%s' for %s aggregation.", selected_region, prefix)

    base = clean.with_columns(
        pl.sum_horizontal([pl.col(c) for c in pos_cols]).alias(f"{prefix}_mw_pos_row"),
        pl.sum_horizontal([pl.col(c) for c in neg_cols]).alias(f"{prefix}_mw_neg_row"),
    ).select(["timestamp_utc", f"{prefix}_mw_pos_row", f"{prefix}_mw_neg_row"])

    # Ensure one row per timestamp before outer joins.
    # - If Germany total columns are used, take max per timestamp
    #   (defensive against repeated identical Germany rows).
    # - Otherwise sum rows per timestamp (typical per-TSO layout).
    if use_de_cols:
        out = (
            base.group_by("timestamp_utc")
            .agg(
                pl.col(f"{prefix}_mw_pos_row").max().alias(f"{prefix}_mw_pos"),
                pl.col(f"{prefix}_mw_neg_row").max().alias(f"{prefix}_mw_neg"),
            )
            .sort("timestamp_utc")
        )
    else:
        out = (
            base.group_by("timestamp_utc")
            .agg(
                pl.col(f"{prefix}_mw_pos_row").sum().alias(f"{prefix}_mw_pos"),
                pl.col(f"{prefix}_mw_neg_row").sum().alias(f"{prefix}_mw_neg"),
            )
            .sort("timestamp_utc")
        )
    return out


def _chunk_range(start: datetime, end: datetime, days: int):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


def _combine_qs_and_operational(
    qs_df: pl.DataFrame,
    op_df: pl.DataFrame,
    base_col: str,
) -> pl.DataFrame:
    """Keep QS/operational raw columns and create canonical + source columns."""
    qs_col = f"{base_col}_qs"
    op_col = f"{base_col}_op"
    source_col = f"{base_col}_source"
    fallback_col = f"{base_col}_source_is_fallback"
    if base_col not in qs_df.columns:
        return qs_df

    left = qs_df.rename({base_col: qs_col})
    right = op_df.rename({base_col: op_col}) if base_col in op_df.columns else pl.DataFrame()
    if right.is_empty():
        merged = left.with_columns(pl.lit(None).cast(pl.Float64).alias(op_col))
    else:
        merged = left.join(
            right.select(["timestamp_utc", op_col]),
            on="timestamp_utc",
            how="full",
            coalesce=True,
        )

    if "timestamp_utc_right" in merged.columns:
        merged = (
            merged.with_columns(
                pl.coalesce([pl.col("timestamp_utc"), pl.col("timestamp_utc_right")]).alias("timestamp_utc")
            )
            .drop("timestamp_utc_right")
        )

    result = (
        merged.with_columns(
            [
                pl.coalesce([pl.col(qs_col), pl.col(op_col)]).alias(base_col),
                pl.when(pl.col(qs_col).is_not_null())
                .then(pl.lit("qs"))
                .when(pl.col(op_col).is_not_null())
                .then(pl.lit("betrieblich"))
                .otherwise(pl.lit("missing"))
                .alias(source_col),
            ]
        )
        .with_columns((pl.col(source_col) == pl.lit("betrieblich")).alias(fallback_col))
        .sort("timestamp_utc")
    )
    return result


def fetch_and_merge(
    start: str,
    end: str,
    token: str,
    timeout: int = 60,
    chunk_days: int = 180,
    chunk_sleep: float = 0.5,
    resample_every: str | None = None,
) -> pl.DataFrame:
    base_url = "https://ds.netztransparenz.de/api/v1/data"
    session = _make_session(token)
    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))

    def _fmt_dt(dt: datetime) -> str:
        # Netztransparenz expects UTC timestamps like 2020-12-01T00:00Z
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")

    frames = []
    def _fetch_chunk_data(s_dt: datetime, e_dt: datetime, retries: int = 2) -> pl.DataFrame | None:
        s_str = _fmt_dt(s_dt)
        e_str = _fmt_dt(e_dt)
        urls = {
            "nrv": f"{base_url}/NrvSaldo/NRVSaldo/Qualitaetsgesichert/{s_str}/{e_str}",
            "rebap": f"{base_url}/NrvSaldo/reBAP/Qualitaetsgesichert/{s_str}/{e_str}",
            "rz": f"{base_url}/NrvSaldo/RZSaldo/Qualitaetsgesichert/{s_str}/{e_str}",
            "afrr": f"{base_url}/NrvSaldo/AktivierteSRL/Qualitaetsgesichert/{s_str}/{e_str}",
            "mfrr": f"{base_url}/NrvSaldo/AktivierteMRL/Qualitaetsgesichert/{s_str}/{e_str}",
        }
        urls_operational = {
            "nrv": f"{base_url}/NrvSaldo/NRVSaldo/Betrieblich/{s_str}/{e_str}",
            "rz": f"{base_url}/NrvSaldo/RZSaldo/Betrieblich/{s_str}/{e_str}",
        }
        try:
            nrv_qs_df = _tidy_nrv(_read_csv(_fetch_csv(urls["nrv"], session, timeout=timeout)))
            rebap_df = _tidy_rebap(_read_csv(_fetch_csv(urls["rebap"], session, timeout=timeout)))
            rz_qs_df = _tidy_rz_saldo(_read_csv(_fetch_csv(urls["rz"], session, timeout=timeout)))
            try:
                rz_op_df = _tidy_rz_saldo(_read_csv(_fetch_csv(urls_operational["rz"], session, timeout=timeout)))
                nrv_op_df = _tidy_nrv(_read_csv(_fetch_csv(urls_operational["nrv"], session, timeout=timeout)))
            except Exception as exc:
                LOGGER.warning("Operational fetch failed for %s -> %s: %s", s_str, e_str, exc)
                rz_op_df = rz_qs_df.select(["timestamp_utc"]).with_columns(
                    pl.lit(None).cast(pl.Float64).alias("rz_saldo_mw")
                ).head(0)
                nrv_op_df = nrv_qs_df.select(["timestamp_utc"]).with_columns(
                    pl.lit(None).cast(pl.Float64).alias("NRV_balance")
                ).head(0)

            rz_df = _combine_qs_and_operational(rz_qs_df, rz_op_df, "rz_saldo_mw")
            nrv_df = _combine_qs_and_operational(nrv_qs_df, nrv_op_df, "NRV_balance")
            rz_nulls = rz_df.select(pl.col("rz_saldo_mw").null_count()).item()
            nrv_nulls = nrv_df.select(pl.col("NRV_balance").null_count()).item()
            LOGGER.debug(
                "RZ/NRV merged %s -> %s (nulls rz=%s nrv=%s).",
                s_str,
                e_str,
                rz_nulls,
                nrv_nulls,
            )
            afrr_df = _tidy_activation(_read_csv(_fetch_csv(urls["afrr"], session, timeout=timeout)), "afrr_activated")
            mfrr_df = _tidy_activation(_read_csv(_fetch_csv(urls["mfrr"], session, timeout=timeout)), "mfrr_activated")
            merged = (
                nrv_df.join(rebap_df, on="timestamp_utc", how="full", coalesce=True)
                .join(rz_df, on="timestamp_utc", how="full", coalesce=True)
                .join(afrr_df, on="timestamp_utc", how="full", coalesce=True)
                .join(mfrr_df, on="timestamp_utc", how="full", coalesce=True)
                .sort("timestamp_utc")
            )
            LOGGER.info("Fetched chunk %s -> %s: %s rows", s_str, e_str, len(merged))
            return merged
        except Exception as exc:
            if retries > 0:
                LOGGER.warning("Chunk %s -> %s failed (%s). Retrying with smaller window.", s_str, e_str, exc)
                mid = s_dt + (e_dt - s_dt) / 2
                left = _fetch_chunk_data(s_dt, mid, retries=retries - 1)
                right = _fetch_chunk_data(mid, e_dt, retries=retries - 1)
                parts = [p for p in [left, right] if p is not None and not p.is_empty()]
                if parts:
                    return pl.concat(parts)
            delta_days = (e_dt - s_dt).days
            if delta_days > 30:
                mid = s_dt + (e_dt - s_dt) / 2
                left = _fetch_chunk_data(s_dt, mid, retries=0)
                right = _fetch_chunk_data(mid, e_dt, retries=0)
                parts = [p for p in [left, right] if p is not None and not p.is_empty()]
                if parts:
                    return pl.concat(parts)
            LOGGER.warning("Chunk %s -> %s failed: %s", s_str, e_str, exc)
            return None

    for s_dt, e_dt in _chunk_range(start_dt, end_dt, chunk_days):
        merged = _fetch_chunk_data(s_dt, e_dt)
        if merged is not None and not merged.is_empty():
            frames.append(merged)
        if chunk_sleep and e_dt < end_dt:
            time.sleep(chunk_sleep)

    if not frames:
        return pl.DataFrame()

    merged = pl.concat(frames).unique(subset=["timestamp_utc"], keep="last").sort("timestamp_utc")

    # Drop duplicate metadata columns from the right-hand frame if they exist.
    drop_cols = [c for c in ("Datum_right", "Zeitzone_right", "von_right", "time_right", "datum_right", "zeitzone_right") if c in merged.columns]
    if drop_cols:
        merged = merged.drop(drop_cols)

    # If Germany total is missing, sum TSO components where available.
    tso_sum_cols = [c for c in ["50hertz", "amprion", "tennet tso", "transnetbw"] if c in merged.columns]
    if tso_sum_cols and "rz_saldo_mw" in merged.columns:
        tso_sum = pl.sum_horizontal([pl.col(c) for c in tso_sum_cols])
        merged = merged.with_columns(
            pl.coalesce([pl.col("rz_saldo_mw"), tso_sum]).alias("rz_saldo_mw")
        )
        if "NRV_balance" in merged.columns:
            merged = merged.with_columns(
                pl.coalesce([pl.col("NRV_balance"), tso_sum]).alias("NRV_balance")
            )

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
        "50hertz",
        "amprion",
        "tennet tso",
        "transnetbw",
    ]
    merged = merged.drop([c for c in tso_cols if c in merged.columns])

    # Keep only one datetime column for merging; drop redundant date/time fields if present.
    merged = merged.drop([c for c in ("Datum", "Zeitzone", "time", "datum", "zeitzone") if c in merged.columns])
    if "nrv_imbalance" in merged.columns and "NRV_balance" in merged.columns:
        merged = merged.drop("nrv_imbalance")

    if resample_every:
        # Downsample numeric values to the target grid using mean aggregation.
        numeric_cols = [c for c, t in zip(merged.columns, merged.dtypes) if t.is_numeric()]
        agg_exprs = []
        if numeric_cols:
            agg_exprs.append(pl.col(numeric_cols).mean())
        merged = (
            merged.sort("timestamp_utc")
            .group_by_dynamic("timestamp_utc", every=resample_every, closed="left", label="left")
            .agg(agg_exprs)
            .sort("timestamp_utc")
        )

    return merged


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fetch NRV-Saldo and reBAP from Netztransparenz and merge on timestamp.")
    parser.add_argument("--start", required=True, help="Start ISO8601 (UTC).")
    parser.add_argument("--end", required=True, help="End ISO8601 (UTC).")
    parser.add_argument("--token", help="Bearer token (defaults to NETZTRANSPARENZ_TOKEN env var).")
    parser.add_argument("--out", default="data/raw/netztransparenz.parquet", help="Output parquet path.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds per request.")
    parser.add_argument("--chunk-days", type=int, default=30, help="Chunk size in days to avoid server errors.")
    parser.add_argument("--chunk-sleep", type=float, default=3.0, help="Seconds to sleep between chunks.")
    parser.add_argument("--resample", default="1h", help="Optional resample frequency (e.g. 1h). Use 'none' to disable.")
    args = parser.parse_args()

    _load_env()
    token = args.token or os.getenv("NETZTRANSPARENZ_TOKEN")
    if not token:
        client_id = os.getenv("NETZTRANSPARENZ_CLIENT_ID")
        client_secret = os.getenv("NETZTRANSPARENZ_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError("No token provided. Set NETZTRANSPARENZ_TOKEN or NETZTRANSPARENZ_CLIENT_ID/NETZTRANSPARENZ_CLIENT_SECRET.")
        token = _fetch_token_from_client_credentials(client_id, client_secret)

    start_utc = datetime.fromisoformat(args.start)
    end_utc = datetime.fromisoformat(args.end)
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=timezone.utc)
    else:
        start_utc = start_utc.astimezone(timezone.utc)
    if end_utc.tzinfo is None:
        end_utc = end_utc.replace(tzinfo=timezone.utc)
    else:
        end_utc = end_utc.astimezone(timezone.utc)

    start_local = start_utc.astimezone(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%dT%H:%M:%S")
    end_local = end_utc.astimezone(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%dT%H:%M:%S")

    resample_every = None if str(args.resample).lower() == "none" else args.resample
    df = fetch_and_merge(
        start_local,
        end_local,
        token,
        timeout=args.timeout,
        chunk_days=args.chunk_days,
        chunk_sleep=args.chunk_sleep,
        resample_every=resample_every,
    )
    if df.is_empty():
        LOGGER.warning("No data fetched.")
        return

    # Sanitize output: UTC hourly and clip to requested window.
    df = df.with_columns(pl.col("timestamp_utc").dt.truncate("1h").alias("timestamp_utc"))
    df = df.filter(
        (pl.col("timestamp_utc") >= pl.lit(start_utc).cast(pl.Datetime(time_unit="us", time_zone="UTC")))
        & (pl.col("timestamp_utc") <= pl.lit(end_utc).cast(pl.Datetime(time_unit="us", time_zone="UTC")))
    ).sort("timestamp_utc")

    if "rz_saldo_mw" in df.columns and df.select(pl.col("rz_saldo_mw").null_count()).item() > 0:
        smard_df = fetch_smard_saldo(start_utc, end_utc)
        if smard_df.is_empty():
            LOGGER.warning("SMARD fallback returned no RZ saldo data.")
        else:
            df = df.join(smard_df, on="timestamp_utc", how="left", suffix="_smard")
            df = df.with_columns(
                pl.coalesce([pl.col("rz_saldo_mw"), pl.col("rz_saldo_mw_smard")]).alias("rz_saldo_mw")
            )
            if "NRV_balance" in df.columns:
                df = df.with_columns(
                    pl.coalesce([pl.col("NRV_balance"), pl.col("rz_saldo_mw_smard")]).alias("NRV_balance")
                )
            df = df.drop("rz_saldo_mw_smard")
            LOGGER.info("Filled RZ saldo nulls from SMARD fallback.")

    # Data quality logs removed per request.

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    LOGGER.info("Wrote %s rows to %s", len(df), out_path)


if __name__ == "__main__":
    main()
