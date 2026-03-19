"""Fetch Regelleistung aFRR CAPACITY/ENERGY results and store as parquet.

Usage:
    ./.venv/bin/python -m energy_trading.ingestion.fetch_regelleistung \
        --start 2020-11-30T23:00:00Z \
        --end 2025-12-31T23:00:00Z \
        --out data/raw/regelleistung.parquet \
        --mol-dir data/raw \
        --bids-dir data/raw/bids

    # Skip bid-based activation reconstruction (faster, no bid files needed)
    ./.venv/bin/python -m energy_trading.ingestion.fetch_regelleistung \
        --start 2020-11-30T23:00:00Z \
        --end 2025-12-31T23:00:00Z \
        --out data/raw/regelleistung.parquet \
        --mol-dir data/raw \
        --skip-bid-activation-prices

Outputs:
    - regelleistung.parquet with hourly aFRR capacity/energy results (UTC, timestamp_utc).

Includes:
    - capacity/energy prices and offered volumes (pos/neg)
    - hourly aFRR VWAP activation prices:
        - afrr_vwap_pos
        - afrr_vwap_neg
    - net_import_export_mw (IGCC/PICASSO netting when present)
    - MOL slope features (from anonymous bid lists, expanded to hourly)
    - hourly anonymous-bid features:
        - afrr_bid_avg_activation_price_{pos,neg}
        - afrr_bid_vwap_activation_price_{pos,neg}
        - bid_alloc_mw_{pos,neg}

Bid file behavior:
    - Tries yearly anonymous energy bid files first (one per year).
    - Falls back to monthly files only if yearly files are not available.
    - Downloads to --bids-dir and reuses existing files.
"""
from __future__ import annotations

import argparse
import calendar
import io
import logging
import re
import time
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Suppress openpyxl style warnings.
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
LOGGER = logging.getLogger(__name__)
CPP_FILES_BASE_URL = "https://www.regelleistung.net/apps/cpp-publisher/api/v2/tenders/files"
# Collect full 15-minute activation price streams while parsing yearly files.
ALL_15MIN_PRICE_DATA: list[pd.DataFrame] = []
PICASSO_START_UTC = pd.Timestamp("2022-06-22 22:00:00+00:00")


def _log_dst_drop(label: str, ts_series: pd.Series, date_series: pd.Series) -> None:
    """Log rows lost due to ambiguous/nonexistent local times during tz conversion."""
    drop_mask = ts_series.isna()
    dropped = int(drop_mask.sum())
    if dropped == 0:
        return
    dates = pd.to_datetime(date_series[drop_mask], errors="coerce").dt.strftime("%Y-%m-%d")
    top = dates.value_counts().head(5)
    top_txt = ", ".join([f"{d}:{int(n)}" for d, n in top.items()]) if len(top) else "n/a"
    LOGGER.warning(
        "DST/timestamp conversion dropped %s rows in %s (top dates: %s).",
        dropped,
        label,
        top_txt,
    )


def _find_col(df: pd.DataFrame, needles: list[str]) -> str | None:
    for c in df.columns:
        c_up = str(c).upper()
        if all(n.upper() in c_up for n in needles):
            return c
    return None


def _parse_product_start(product: str) -> int | None:
    """Parse aFRR product like NEG_00_04 into start hour."""
    try:
        parts = str(product).split("_")
        if len(parts) >= 2:
            return int(parts[1])
    except Exception:
        return None
    return None


def _parse_date_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")
    serial = pd.to_numeric(series, errors="coerce")
    # Some XLSX readers expose Excel serials (e.g. 45658) instead of datetimes.
    use_serial = serial.notna() & (parsed.isna() | (parsed.dt.year < 1995))
    if use_serial.any():
        parsed.loc[use_serial] = pd.to_datetime(
            serial.loc[use_serial], unit="D", origin="1899-12-30", errors="coerce"
        )
    return parsed


def _normalize_col_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _normalize_regelleistung_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy German headers to standard English Regelleistung columns."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    norm_map = {_normalize_col_name(str(c)): c for c in df.columns}

    # Map legacy German headers to standard English names.
    mappings: dict[str, list[str]] = {
        "GERMANY_AVERAGE_ENERGY_PRICE_[EUR/MWh]": [
            "mittelwertarbeitspreis",
            "mittelwert_arbeitspreis",
            "mittelwertarbeitspreis[eur/mwh]",
        ],
        "GERMANY_MARGINAL_ENERGY_PRICE_[EUR/MWh]": [
            "grenzpreis",
            "grenzpreis[eur/mwh]",
        ],
        "GERMANY_MIN_ENERGY_PRICE_[EUR/MWh]": [
            "minarbeitspreis",
            "min_arbeitspreis",
            "minarbeitspreis[eur/mwh]",
        ],
        "GERMANY_SUM_OF_OFFERED_CAPACITY_[MW]": [
            "angeboteneleistung",
            "angebotene_leistung",
            "angeboteneleistung[mw]",
        ],
    }

    for target, variants in mappings.items():
        if target in df.columns:
            continue
        for variant in variants:
            col = norm_map.get(_normalize_col_name(variant))
            if col and col in df.columns:
                df = df.rename(columns={col: target})
                break
    return df


def _mol_slope_from_bids(df_bids: pd.DataFrame, market: str) -> pd.DataFrame:
    """Compute MOL slope P500-P100 per Date+Product+ReserveType."""
    df_bids.columns = df_bids.columns.str.strip()
    date_col = _find_col(df_bids, ["DELIVERY", "DATE"]) or _find_col(df_bids, ["DATUM"]) or df_bids.columns[0]
    prod_col = _find_col(df_bids, ["PRODUCT"]) or _find_col(df_bids, ["PRODUKT"]) or df_bids.columns[1]
    res_col = _find_col(df_bids, ["RESERVE"]) or _find_col(df_bids, ["RESERVETYPE"])
    cap_col = _find_col(df_bids, ["OFFERED", "CAPACITY"]) or _find_col(df_bids, ["CAPACITY", "MW"])
    if market == "energy":
        price_col = _find_col(df_bids, ["ENERGY", "PRICE"]) or _find_col(df_bids, ["PRICE"])
    else:
        price_col = _find_col(df_bids, ["CAPACITY", "PRICE"]) or _find_col(df_bids, ["PRICE"])
    if not price_col or not cap_col:
        return pd.DataFrame()

    keep = [date_col, prod_col, price_col, cap_col]
    if res_col:
        keep.append(res_col)
    df_bids = df_bids[keep].copy()
    df_bids[date_col] = _parse_date_series(df_bids[date_col])
    df_bids[price_col] = pd.to_numeric(df_bids[price_col], errors="coerce")
    df_bids[cap_col] = pd.to_numeric(df_bids[cap_col], errors="coerce")
    df_bids = df_bids.dropna(subset=[date_col, prod_col, price_col, cap_col])

    group_cols = [date_col, prod_col] + ([res_col] if res_col else [])
    rows = []
    for keys, g in df_bids.groupby(group_cols):
        g = g.sort_values(price_col)
        g["cum"] = g[cap_col].cumsum()
        total = g["cum"].iloc[-1]
        if total < 100:
            slope = float("nan")
        else:
            # idxmax returns the original index label, so we must use .loc (not .iloc).
            p100 = (
                float(g.loc[(g["cum"] >= 100).idxmax(), price_col])
                if (g["cum"] >= 100).any()
                else float(g[price_col].max())
            )
            if total < 500:
                p500 = float(g[price_col].max())
            else:
                p500 = float(g.loc[(g["cum"] >= 500).idxmax(), price_col])
            slope = p500 - p100
        if res_col:
            d, prod, reserve = keys
        else:
            d, prod = keys
            reserve = "aFRR"
        rows.append((d, prod, reserve, slope))

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["date", "product", "reserve_type", "mol_slope"])


def _expand_energy_mol(df: pd.DataFrame) -> pd.DataFrame:
    """Energy market: 15-min products like NEG_001 -> map to hour and average per hour."""
    if df.empty:
        return df
    prod_num = df["product"].str.extract(r"(\\d+)").astype(float)
    df = df.assign(prod_num=prod_num)
    df = df.dropna(subset=["prod_num"])
    df["hour"] = ((df["prod_num"] - 1) // 4).astype(int)
    df["timestamp"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
    df = df.dropna(subset=["timestamp"])
    out = (
        df.groupby(["timestamp", "reserve_type"])["mol_slope"]
        .mean()
        .unstack("reserve_type")
        .add_prefix("energy_mol_slope_")
        .reset_index()
    )
    return out


def _expand_capacity_mol(df: pd.DataFrame) -> pd.DataFrame:
    """Capacity market: 4-hour blocks like NEG_00_04 -> ffill across block hours."""
    if df.empty:
        return df
    start_hour = df["product"].str.extract(r"_(\\d{2})_").astype(float)
    df = df.assign(start_hour=start_hour).dropna(subset=["start_hour"])
    start_local = pd.to_datetime(df["date"]).dt.tz_localize(
        "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward"
    )
    df["timestamp"] = start_local + pd.to_timedelta(df["start_hour"].astype(int), unit="h")
    df = df.dropna(subset=["timestamp"])
    rows = []
    for _, r in df.iterrows():
        block = pd.date_range(
            start=r["timestamp"],
            periods=4,
            freq="1h",
            inclusive="left",
        )
        for ts in block:
            rows.append((ts, r["reserve_type"], r["mol_slope"]))
    out = pd.DataFrame(rows, columns=["timestamp", "reserve_type", "mol_slope"])
    out = (
        out.groupby(["timestamp", "reserve_type"])["mol_slope"]
        .mean()
        .unstack("reserve_type")
        .add_prefix("capacity_mol_slope_")
        .reset_index()
    )
    return out


def process_mol_slope(data_dir: Path) -> pd.DataFrame:
    """Process anonymous bid files into hourly MOL slope features."""
    patterns = [
        ("energy", "RESULT_LIST_ANONYM_ENERGY_MARKET_*.zip"),
        ("capacity", "RESULT_LIST_ANONYM_CAPACITY_MARKET_*.zip"),
        ("energy", "RESULT_LIST_ANONYM_ENERGY_MARKET_*.xlsx"),
        ("capacity", "RESULT_LIST_ANONYM_CAPACITY_MARKET_*.xlsx"),
    ]
    dfs_energy = []
    dfs_capacity = []
    for market, pat in patterns:
        for path in data_dir.rglob(pat):
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path, "r") as zf:
                    for name in zf.namelist():
                        if name.lower().endswith((".xlsx", ".xls")):
                            with zf.open(name) as f:
                                df = pd.read_excel(f, engine="openpyxl")
                                mol = _mol_slope_from_bids(df, market=market)
                                if market == "energy":
                                    dfs_energy.append(mol)
                                else:
                                    dfs_capacity.append(mol)
            else:
                df = pd.read_excel(path, engine="openpyxl")
                mol = _mol_slope_from_bids(df, market=market)
                if market == "energy":
                    dfs_energy.append(mol)
                else:
                    dfs_capacity.append(mol)

    energy = _expand_energy_mol(pd.concat(dfs_energy, ignore_index=True)) if dfs_energy else pd.DataFrame()
    capacity = _expand_capacity_mol(pd.concat(dfs_capacity, ignore_index=True)) if dfs_capacity else pd.DataFrame()

    if energy.empty and capacity.empty:
        return pd.DataFrame()

    out = energy if capacity.empty else energy.merge(capacity, on="timestamp", how="outer")
    ts = pd.to_datetime(out["timestamp"], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="NaT")
    ts = ts.dt.tz_convert("UTC")
    out["timestamp"] = ts
    out = out.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    return out


def _bid_prices_from_bids(df_bids: pd.DataFrame) -> pd.DataFrame:
    """Extract hourly bid-based price features from anonymous aFRR energy bids."""
    if df_bids.empty:
        return pd.DataFrame()
    df = df_bids.copy()
    df.columns = df.columns.str.strip()

    date_col = _find_col(df, ["DELIVERY", "DATE"]) or _find_col(df, ["DATUM"]) or df.columns[0]
    prod_col = _find_col(df, ["PRODUCT"]) or _find_col(df, ["PRODUKT"])
    reserve_col = _find_col(df, ["RESERVE"]) or _find_col(df, ["RESERVETYPE"])
    alloc_col = _find_col(df, ["ALLOCATED", "CAPACITY"]) or _find_col(df, ["CAPACITY", "MW"])
    price_col = _find_col(df, ["ENERGY", "PRICE"]) or _find_col(df, ["PRICE"])
    pay_col = _find_col(df, ["PAYMENT", "DIRECTION"])
    country_col = _find_col(df, ["COUNTRY"])

    if not prod_col or not alloc_col or not price_col:
        return pd.DataFrame()

    keep = [date_col, prod_col, alloc_col, price_col]
    if reserve_col:
        keep.append(reserve_col)
    if pay_col:
        keep.append(pay_col)
    if country_col:
        keep.append(country_col)
    df = df[keep].copy()

    df[date_col] = _parse_date_series(df[date_col])
    df[alloc_col] = pd.to_numeric(df[alloc_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[date_col, alloc_col, price_col])
    df = df[df[alloc_col] > 0]

    if reserve_col:
        df = df[df[reserve_col].astype(str).str.upper().str.contains("AFRR", na=False)]
    if country_col:
        df = df[df[country_col].astype(str).str.upper().eq("DE")]

    prod = df[prod_col].astype(str)
    df["direction"] = prod.str.split("_").str[0].str.upper()
    df = df[df["direction"].isin(["POS", "NEG"])]

    # PRODUCT has two formats in source files:
    #   1) NEG_001 .. NEG_096 (quarter-hour index)
    #   2) NEG_04_08 (hour block, start-end)
    qh = pd.to_numeric(prod.str.extract(r"_(\d{3})$")[0], errors="coerce")
    start_hour = pd.to_numeric(prod.str.extract(r"_(\d{2})_(\d{2})$")[0], errors="coerce")
    end_hour = pd.to_numeric(prod.str.extract(r"_(\d{2})_(\d{2})$")[1], errors="coerce")
    mask_qh = qh.notna()
    mask_block = start_hour.notna() & end_hour.notna()

    parts: list[pd.DataFrame] = []

    if mask_qh.any():
        q = df.loc[mask_qh].copy()
        qh_q = qh.loc[mask_qh].astype(int)
        base_local = pd.to_datetime(q[date_col]).dt.floor("D").dt.tz_localize(
            "Europe/Berlin", ambiguous="NaT", nonexistent="shift_forward"
        )
        q_ts = (
            base_local + pd.to_timedelta((qh_q - 1) * 15, unit="m")
        ).dt.tz_convert("UTC")
        _log_dst_drop("bid qh-format", q_ts, q[date_col])
        q["timestamp_utc"] = q_ts
        parts.append(q)

    if mask_block.any():
        b = df.loc[mask_block].copy()
        b["start_hour"] = start_hour.loc[mask_block].astype(int)
        b["end_hour"] = end_hour.loc[mask_block].astype(int)
        b["hour_offsets"] = [
            list(range(sh, eh if eh > sh else eh + 24))
            for sh, eh in zip(b["start_hour"], b["end_hour"])
        ]
        b = b.explode("hour_offsets")
        base_local = pd.to_datetime(b[date_col]).dt.floor("D").dt.tz_localize(
            "Europe/Berlin", ambiguous="NaT", nonexistent="shift_forward"
        )
        b_ts = (
            base_local + pd.to_timedelta(b["hour_offsets"].astype(int), unit="h")
        ).dt.tz_convert("UTC")
        _log_dst_drop("bid block-format", b_ts, b[date_col])
        b["timestamp_utc"] = b_ts
        b = b.drop(columns=["start_hour", "end_hour", "hour_offsets"])
        parts.append(b)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["timestamp_utc"])

    pay_sign = 1.0
    if pay_col:
        pay = df[pay_col].astype(str).str.upper()
        pay_sign = np.where(pay.str.contains("PROVIDER_TO_GRID", na=False), -1.0, 1.0)
    dir_sign = np.where(df["direction"].eq("NEG"), -1.0, 1.0)
    df["price_signed"] = df[price_col] * pay_sign * dir_sign
    df["pv"] = df["price_signed"] * df[alloc_col]
    df["hour"] = df["timestamp_utc"].dt.floor("1h")

    g = (
        df.groupby(["hour", "direction"], as_index=False)
        .agg(
            bid_vwap_num=("pv", "sum"),
            bid_alloc_mw=(alloc_col, "sum"),
            bid_price_max=("price_signed", "max"),
            bid_price_min=("price_signed", "min"),
        )
    )
    g["afrr_bid_vwap_activation_price"] = np.where(
        g["bid_alloc_mw"] != 0, g["bid_vwap_num"] / g["bid_alloc_mw"], np.nan
    )
    # Marginal proxy: POS -> max accepted signed price, NEG -> min accepted signed price.
    g["afrr_bid_avg_activation_price"] = np.where(
        g["direction"].eq("POS"), g["bid_price_max"], g["bid_price_min"]
    )

    out = g.pivot_table(
        index="hour",
        columns="direction",
        values=["afrr_bid_vwap_activation_price", "afrr_bid_avg_activation_price", "bid_alloc_mw"],
        aggfunc="mean",
    )
    out.columns = [f"{m}_{d.lower()}" for m, d in out.columns]
    out = out.rename_axis("timestamp").sort_index()
    return out


def _bid_usecol(name: str) -> bool:
    """Keep only columns needed for bid-based price reconstruction."""
    n = _normalize_col_name(str(name))
    keywords = (
        "deliverydate",
        "datum",
        "product",
        "produkt",
        "typeofreserves",
        "reservetype",
        "country",
        "allocatedcapacity",
        "capacitymw",
        "energyprice",
        "price",
        "paymentdirection",
    )
    return any(k in n for k in keywords)


def _file_signature(path: Path) -> str:
    st = path.stat()
    return f"{path.name}|{st.st_size}|{int(st.st_mtime)}"


def process_bid_prices(data_dir: Path) -> pd.DataFrame:
    """Process anonymous energy bids into hourly price features."""
    raw_files = sorted(
        p
        for p in data_dir.rglob("RESULT_LIST_ANONYM_ENERGY_MARKET_*.xlsx*")
        if p.is_file() and not p.name.startswith("~$")
    )
    period_re = re.compile(
        r"RESULT_LIST_ANONYM_ENERGY_MARKET_aFRR_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.xlsx(?:\.zip)?$",
        flags=re.IGNORECASE,
    )
    selected: dict[str, Path] = {}
    for p in raw_files:
        m = period_re.search(p.name)
        key = f"{m.group(1)}_{m.group(2)}" if m else p.stem.lower()
        # Prefer zipped source when both .xlsx and .xlsx.zip exist.
        if key not in selected or p.suffix.lower() == ".zip":
            selected[key] = p
    files = sorted(selected.values())
    LOGGER.info(
        "Anonymous bid files selected for marginal-price calculation: %s (from %s discovered)",
        len(files),
        len(raw_files),
    )
    if not files:
        return pd.DataFrame()

    cache_path = data_dir / "_afrr_bid_hourly_cache.parquet"
    cached = pd.DataFrame()
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            required = {
                "source_file",
                "file_sig",
                "timestamp",
                "afrr_bid_vwap_activation_price_pos",
                "afrr_bid_vwap_activation_price_neg",
                "afrr_bid_avg_activation_price_pos",
                "afrr_bid_avg_activation_price_neg",
                "bid_alloc_mw_pos",
                "bid_alloc_mw_neg",
            }
            if not required.issubset(set(cached.columns)):
                LOGGER.warning("Bid cache schema mismatch; ignoring %s", cache_path)
                cached = pd.DataFrame()
        except Exception as exc:
            LOGGER.warning("Failed to read bid cache %s: %s", cache_path, exc)
            cached = pd.DataFrame()

    frames_with_meta: list[pd.DataFrame] = []
    cached_hits = 0
    parsed_files = 0
    t0 = time.perf_counter()
    for path in files:
        sig = _file_signature(path)
        if not cached.empty:
            hit = cached[(cached["source_file"] == path.name) & (cached["file_sig"] == sig)]
            if not hit.empty:
                cached_hits += 1
                frames_with_meta.append(hit.copy())
                LOGGER.info("Using bid cache for %s: %s hourly rows", path.name, len(hit))
                continue

        try:
            per_file_parts: list[pd.DataFrame] = []
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path, "r") as zf:
                    for name in zf.namelist():
                        if name.lower().endswith((".xlsx", ".xls")):
                            with zf.open(name) as f:
                                xls = pd.ExcelFile(f, engine="openpyxl")
                                for sheet in xls.sheet_names:
                                    df = xls.parse(sheet_name=sheet, usecols=_bid_usecol)
                                    x = _bid_prices_from_bids(df)
                                    if not x.empty:
                                        per_file_parts.append(x)
                                file_rows = sum(len(x) for x in per_file_parts)
                                LOGGER.info(
                                    "Parsed bid marginal prices from %s/%s (%s sheets): %s hourly rows",
                                    path.name,
                                    name,
                                    len(xls.sheet_names),
                                    file_rows,
                                )
            else:
                xls = pd.ExcelFile(path, engine="openpyxl")
                for sheet in xls.sheet_names:
                    df = xls.parse(sheet_name=sheet, usecols=_bid_usecol)
                    x = _bid_prices_from_bids(df)
                    if not x.empty:
                        per_file_parts.append(x)
                file_rows = sum(len(x) for x in per_file_parts)
                LOGGER.info(
                    "Parsed bid marginal prices from %s (%s sheets): %s hourly rows",
                    path.name,
                    len(xls.sheet_names),
                    file_rows,
                )
            if per_file_parts:
                parsed_files += 1
                per_file = pd.concat(per_file_parts).sort_index()
                per_file = per_file.groupby(level=0).mean(numeric_only=True)
                per_file = per_file.reset_index().rename(columns={"index": "timestamp"})
                per_file["source_file"] = path.name
                per_file["file_sig"] = sig
                frames_with_meta.append(per_file)
        except Exception as exc:  # pragma: no cover - defensive parsing
            LOGGER.warning("Failed to parse bid file %s: %s", path, exc)

    if not frames_with_meta:
        LOGGER.warning("No usable anonymous bid data for marginal-price calculation.")
        return pd.DataFrame()

    combined = pd.concat(frames_with_meta, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["timestamp"])
    try:
        combined.to_parquet(cache_path, index=False)
        LOGGER.info(
            "Updated bid cache: %s rows (%s cache hits, %s parsed files) in %.1fs",
            len(combined),
            cached_hits,
            parsed_files,
            time.perf_counter() - t0,
        )
    except Exception as exc:
        LOGGER.warning("Failed to write bid cache %s: %s", cache_path, exc)

    out = combined.drop(columns=["source_file", "file_sig"], errors="ignore").set_index("timestamp").sort_index()
    raw_rows = len(out)
    out = out.groupby(level=0).mean(numeric_only=True)
    out = out[~out.index.duplicated(keep="last")]
    LOGGER.info(
        "Computed anonymous-bid hourly prices: rows=%s (from %s pre-agg rows), range=%s -> %s",
        len(out),
        raw_rows,
        out.index.min(),
        out.index.max(),
    )
    for col in (
        "afrr_bid_avg_activation_price_pos",
        "afrr_bid_avg_activation_price_neg",
        "afrr_bid_vwap_activation_price_pos",
        "afrr_bid_vwap_activation_price_neg",
    ):
        if col in out.columns:
            LOGGER.info("Non-null %s: %s", col, int(out[col].notna().sum()))
    return out


def _download_bid_result_file(period_start: str, period_end: str, out_dir: Path) -> Path | None:
    """Try to download one anonymous bid result file for the given period."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"RESULT_LIST_ANONYM_ENERGY_MARKET_aFRR_{period_start}_{period_end}.xlsx"
    candidates = [f"{base}.zip", base]
    for name in candidates:
        target = out_dir / name
        if target.exists() and target.stat().st_size > 0:
            LOGGER.info("Bid file already exists: %s", target.name)
            return target
        url = f"{CPP_FILES_BASE_URL}/{name}"
        try:
            resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            target.write_bytes(resp.content)
            LOGGER.info("Downloaded bid file: %s", target.name)
            return target
        except Exception:
            continue
    return None


def download_bid_files_with_fallback(start_dt: datetime, end_dt: datetime, bids_dir: Path) -> list[Path]:
    """Download yearly anonymous bid files; if missing, fallback to monthly files."""
    downloaded: list[Path] = []
    for year in range(start_dt.year, end_dt.year + 1):
        year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
        year_end = datetime(year, 12, 31, tzinfo=timezone.utc)
        if year_end < start_dt or year_start > end_dt:
            continue

        yearly = _download_bid_result_file(f"{year}-01-01", f"{year}-12-31", bids_dir)
        if yearly is not None:
            downloaded.append(yearly)
            continue

        LOGGER.warning("Yearly anonymous bid file missing for %s. Falling back to monthly files.", year)
        for month in range(1, 13):
            month_start = datetime(year, month, 1, tzinfo=timezone.utc)
            month_last = calendar.monthrange(year, month)[1]
            month_end = datetime(year, month, month_last, tzinfo=timezone.utc)
            if month_end < start_dt or month_start > end_dt:
                continue
            monthly = _download_bid_result_file(
                f"{year}-{month:02d}-01",
                f"{year}-{month:02d}-{month_last:02d}",
                bids_dir,
            )
            if monthly is not None:
                downloaded.append(monthly)
            else:
                LOGGER.warning("Missing monthly anonymous bid file for %s-%02d.", year, month)
    LOGGER.info("Anonymous bid download complete. Files available/added: %s", len(downloaded))
    return downloaded


def _download_overview_file(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    df = pd.read_excel(io.BytesIO(resp.content), engine="openpyxl")
    return _normalize_regelleistung_columns(df)


def _fetch_monthly_overview(
    year: int,
    market_type: str,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for month in range(1, 13):
        if start_dt or end_dt:
            month_start = datetime(year, month, 1, tzinfo=timezone.utc)
            last_day = calendar.monthrange(year, month)[1]
            month_end = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)
            if start_dt and month_end < start_dt:
                continue
            if end_dt and month_start > end_dt:
                continue
        last_day = calendar.monthrange(year, month)[1]
        url = (
            "https://www.regelleistung.net/apps/cpp-publisher/api/v2/tenders/files/"
            f"RESULT_OVERVIEW_{market_type}_MARKET_aFRR_{year}-{month:02d}-01_{year}-{month:02d}-{last_day:02d}.xlsx"
        )
        try:
            resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except Exception:
            LOGGER.warning("Missing monthly %s %04d-%02d overview file.", market_type, year, month)
            continue
        df = _normalize_regelleistung_columns(pd.read_excel(io.BytesIO(resp.content), engine="openpyxl"))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_and_parse_regelleistung(
    year: int,
    market_type: str,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
) -> pd.DataFrame:
    """Download aFRR results and return a tidy DataFrame indexed by timestamp."""
    url = (
        "https://www.regelleistung.net/apps/cpp-publisher/api/v2/tenders/files/"
        f"RESULT_OVERVIEW_{market_type}_MARKET_aFRR_{year}-01-01_{year}-12-31.xlsx"
    )
    LOGGER.info("Downloading %s data for %s...", market_type, year)
    df = pd.DataFrame()
    try:
        df = _download_overview_file(url)
    except Exception:
        LOGGER.warning("Yearly %s %s overview file not found.", year, market_type)

    # --- 1) Dynamic column detection ---
    date_col = _find_col(df, ["DATE"]) or _find_col(df, ["DATUM"]) or (df.columns[0] if not df.empty else None)
    prod_col = (
        _find_col(df, ["PRODUC"])
        or _find_col(df, ["PRODUKT"])
        or _find_col(df, ["TIME"])
        or (df.columns[1] if not df.empty and len(df.columns) > 1 else None)
    )

    # If this is the start year, accept partial yearly data if it covers the requested window.
    allow_partial = False
    if (
        market_type in {"ENERGY", "CAPACITY"}
        and start_dt is not None
        and year == start_dt.year
        and not df.empty
        and date_col is not None
    ):
        max_date = _parse_date_series(df[date_col]).max()
        if pd.notna(max_date) and max_date.date() >= start_dt.date():
            allow_partial = True
            LOGGER.info("Yearly %s %s accepted as partial coverage (max date %s).", market_type, year, max_date.date())

    # If yearly file is missing/sparse, try monthly fallbacks (ENERGY and CAPACITY).
    min_rows = 4000 if market_type == "ENERGY" else 2000
    if market_type in {"ENERGY", "CAPACITY"} and (df.empty or len(df) < min_rows) and not allow_partial:
        LOGGER.warning(
            "Yearly %s file for %s is missing or sparse (%s rows). Falling back to monthly files.",
            market_type,
            year,
            len(df),
        )
        df = _fetch_monthly_overview(year, market_type, start_dt=start_dt, end_dt=end_dt)

    if df.empty:
        LOGGER.warning("Skipping %s %s: no usable data.", year, market_type)
        return pd.DataFrame()

    # --- 1) Dynamic column detection ---
    date_col = _find_col(df, ["DATE"]) or _find_col(df, ["DATUM"]) or df.columns[0]
    prod_col = _find_col(df, ["PRODUC"]) or _find_col(df, ["PRODUKT"]) or _find_col(df, ["TIME"]) or df.columns[1]

    # --- 2) Timestamp parsing (mixed formats supported) ---
    prod_str = df[prod_col].astype(str)
    mask_block = prod_str.str.contains(r"_\d{2}_\d{2}")
    mask_qh = prod_str.str.contains(r"_\d{3}") & ~mask_block

    def _process_part(df_part: pd.DataFrame, mode: str) -> pd.DataFrame:
        if df_part.empty:
            return pd.DataFrame()

        part_prod = df_part[prod_col].astype(str)
        start_of_day = pd.to_datetime(df_part[date_col]).dt.tz_localize(
            "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward"
        )
        if mode == "block":
            df_part["start_hour"] = part_prod.str.extract(r"_(\d{2})_")
            df_part["start_hour"] = pd.to_numeric(df_part["start_hour"], errors="coerce")
            df_part = df_part.dropna(subset=["start_hour"])
            df_part["start_hour"] = df_part["start_hour"].astype(int)
            df_part["timestamp_local"] = start_of_day + pd.to_timedelta(df_part["start_hour"], unit="h")
            df_part = df_part.dropna(subset=["timestamp_local"])
            df_part["timestamp"] = df_part["timestamp_local"].apply(
                lambda ts: pd.date_range(start=ts, periods=4, freq="1h", inclusive="left")
            )
            df_part = df_part.explode("timestamp").drop(columns=["timestamp_local"])
            df_part["Richtung"] = part_prod.str.split("_").str[0]
        elif mode == "qh":
            df_part["quarter_hour_int"] = part_prod.str.extract(r"_(\d+)").astype(int)
            df_part["time_offset"] = pd.to_timedelta((df_part["quarter_hour_int"] - 1) * 15, unit="m")
            df_part["timestamp"] = start_of_day + df_part["time_offset"]
            df_part["Richtung"] = part_prod.str.split("_").str[0]
        else:
            df_part["start_time"] = df_part[prod_col].astype(str).str.split(" - ").str[0]
            naive = pd.to_datetime(df_part[date_col].astype(str) + " " + df_part["start_time"])
            df_part["timestamp"] = naive.dt.tz_localize(
                "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward"
            )
            dir_col = _find_col(df_part, ["DIRECTION"]) or _find_col(df_part, ["RICHTUNG"]) or df_part.columns[2]
            df_part["Richtung"] = df_part[dir_col]

        df_part["timestamp"] = df_part["timestamp"].dt.tz_convert("UTC")
        df_part = df_part.dropna(subset=["timestamp"])

        # --- 3) Feature columns ---
        val_cols: dict[str, str] = {}
        if market_type == "ENERGY":
            marg_col = _find_col(df_part, ["MARGINAL", "ENERGY", "PRICE"])
            avg_col = _find_col(df_part, ["AVERAGE", "ENERGY", "PRICE"])
            off_col = _find_col(df_part, ["OFFERED", "CAPACITY"])
            if marg_col:
                val_cols[marg_col] = "afrr_marginal_activation_price"
            if avg_col:
                val_cols[avg_col] = "afrr_avg_activation_price"
            if off_col:
                val_cols[off_col] = "afrr_activation_offered_mw"
        else:
            marg_col = _find_col(df_part, ["MARGINAL", "CAPACITY", "PRICE"]) or _find_col(df_part, ["GRENZWERT"])
            off_col = _find_col(df_part, ["OFFERED", "CAPACITY"])
            if marg_col:
                val_cols[marg_col] = "afrr_capacity_price"
            if off_col:
                val_cols[off_col] = "afrr_capacity_offered_mw"

        for col, out_name in val_cols.items():
            series = pd.to_numeric(df_part[col], errors="coerce")
            df_part[col] = series

        # --- 4) European netting (import/export) ---
        net_col = _find_col(df_part, ["GERMANY_IMPORT", "EXPORT"])
        net_series = None
        if net_col:
            net_series = (
                df_part[["timestamp", net_col]]
                .assign(**{net_col: pd.to_numeric(df_part[net_col], errors="coerce")})
                .groupby("timestamp")[net_col]
                .mean()
                .rename("net_import_export_mw")
            )

        # --- 5) Pivot by direction ---
        df_clean = df_part[["timestamp", "Richtung"] + list(val_cols.keys())]
        df_pivot = df_clean.pivot_table(index="timestamp", columns="Richtung", values=list(val_cols.keys()))

        new_cols = []
        for col in df_pivot.columns:
            orig_metric = col[0]
            direction = str(col[1]).replace("ATIVE", "").lower()
            new_cols.append(f"{val_cols[orig_metric]}_{direction}")
        df_pivot.columns = new_cols

        if net_series is not None:
            df_pivot = df_pivot.join(net_series, how="left")

        return df_pivot

    parts = []
    if mask_block.any():
        parts.append(_process_part(df[mask_block].copy(), "block"))
    if mask_qh.any():
        parts.append(_process_part(df[mask_qh].copy(), "qh"))
    if not mask_block.any() and not mask_qh.any():
        parts.append(_process_part(df.copy(), "other"))

    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    df_pivot = pd.concat(parts).sort_index()
    df_pivot = df_pivot[~df_pivot.index.duplicated(keep="first")]
    return df_pivot


def _load_netztransparenz_activation_15m(
    netz_path: Path,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """Load 15-minute aFRR activated volumes from Netztransparenz parquet."""
    if not netz_path.exists():
        LOGGER.warning("Netztransparenz parquet not found for VWAP join: %s", netz_path)
        return pd.DataFrame()

    df = pd.read_parquet(netz_path)
    ts_col = "timestamp_utc" if "timestamp_utc" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    if ts_col is None:
        LOGGER.warning("No timestamp column found in %s for VWAP join.", netz_path)
        return pd.DataFrame()

    required_cols = ["afrr_activated_mw_pos", "afrr_activated_mw_neg"]
    found_cols = [c for c in required_cols if c in df.columns]
    if not found_cols:
        LOGGER.warning("No aFRR activated volume columns found in %s for VWAP join.", netz_path)
        return pd.DataFrame()

    df = df[[ts_col] + found_cols].copy()
    df["timestamp_utc"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    drop_cols = [ts_col] if ts_col != "timestamp_utc" else []
    df = (
        df.dropna(subset=["timestamp_utc"])
        .drop(columns=drop_cols)
        .set_index("timestamp_utc")
        .sort_index()
    )
    df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]

    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    df.index = df.index.floor("15min")
    return df.groupby(df.index).mean(numeric_only=True)[required_cols]


def _aggregate_hourly_with_vwap(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to hourly means and compute aFRR VWAP from 15-minute rows."""
    if df_15m.empty:
        return df_15m

    df_15m = df_15m.copy()
    if df_15m.index.tz is None:
        df_15m.index = df_15m.index.tz_localize("UTC")
    else:
        df_15m.index = df_15m.index.tz_convert("UTC")
    df_15m.index = df_15m.index.floor("15min")
    df_15m = df_15m[~df_15m.index.duplicated(keep="last")].sort_index()

    price_pos = "afrr_avg_activation_price_pos"
    price_neg = "afrr_avg_activation_price_neg"
    vol_pos = "afrr_activated_mw_pos"
    vol_neg = "afrr_activated_mw_neg"

    # Pre-PICASSO prices are often posted only at :00. Fill :15/:30/:45 inside
    # each hour before weighting, but never propagate across hour boundaries.
    pre_picasso_mask = df_15m.index < PICASSO_START_UTC
    if pre_picasso_mask.any():
        if price_pos in df_15m.columns:
            pre_idx = df_15m.index[pre_picasso_mask]
            df_15m.loc[pre_picasso_mask, price_pos] = (
                df_15m.loc[pre_picasso_mask, price_pos]
                .groupby(pre_idx.floor("1h"))
                .ffill()
            )
        if price_neg in df_15m.columns:
            pre_idx = df_15m.index[pre_picasso_mask]
            df_15m.loc[pre_picasso_mask, price_neg] = (
                df_15m.loc[pre_picasso_mask, price_neg]
                .groupby(pre_idx.floor("1h"))
                .ffill()
            )

    if {price_pos, vol_pos}.issubset(df_15m.columns):
        df_15m["weighted_cost_pos"] = df_15m[price_pos] * df_15m[vol_pos].abs()
    if {price_neg, vol_neg}.issubset(df_15m.columns):
        df_15m["weighted_cost_neg"] = df_15m[price_neg] * df_15m[vol_neg].abs()

    hourly = df_15m.resample("1h").mean(numeric_only=True)

    if {"weighted_cost_pos", vol_pos, price_pos}.issubset(df_15m.columns):
        sum_weighted_pos = df_15m["weighted_cost_pos"].resample("1h").sum(min_count=1)
        sum_vol_pos = df_15m[vol_pos].abs().resample("1h").sum(min_count=1)
        mean_price_pos = df_15m[price_pos].resample("1h").mean()
        vwap_pos = np.where(sum_vol_pos != 0, sum_weighted_pos / sum_vol_pos, mean_price_pos)
        hourly["afrr_vwap_pos"] = vwap_pos

    if {"weighted_cost_neg", vol_neg, price_neg}.issubset(df_15m.columns):
        sum_weighted_neg = df_15m["weighted_cost_neg"].resample("1h").sum(min_count=1)
        sum_vol_neg = df_15m[vol_neg].abs().resample("1h").sum(min_count=1)
        mean_price_neg = df_15m[price_neg].resample("1h").mean()
        vwap_neg = np.where(sum_vol_neg != 0, sum_weighted_neg / sum_vol_neg, mean_price_neg)
        hourly["afrr_vwap_neg"] = vwap_neg

    return hourly.drop(columns=[c for c in ["weighted_cost_pos", "weighted_cost_neg"] if c in hourly.columns], errors="ignore")


def _export_afrr_15min_price_volume(
    price_15m: pd.DataFrame,
    volume_15m: pd.DataFrame,
    out_dir: Path,
    start_dt: datetime,
    end_dt: datetime,
) -> None:
    """Persist raw 15-minute aFRR prices and volumes for VWAP validation."""
    if price_15m.empty and volume_15m.empty:
        LOGGER.warning("No 15-minute aFRR price/volume data to export.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    if not price_15m.empty:
        p = price_15m.copy()
        p.index = pd.to_datetime(p.index, utc=True, errors="coerce")
        p = p[~p.index.isna()]
        p = p[(p.index >= start_dt) & (p.index <= end_dt)]
        p.index = p.index.floor("15min")
        p = p[~p.index.duplicated(keep="last")].sort_index()
        frames.append(p)
        p.reset_index(names="timestamp_utc").to_parquet(
            out_dir / "afrr_prices_15min.parquet",
            index=False,
            compression="zstd",
        )

    if not volume_15m.empty:
        v = volume_15m.copy()
        v.index = pd.to_datetime(v.index, utc=True, errors="coerce")
        v = v[~v.index.isna()]
        v = v[(v.index >= start_dt) & (v.index <= end_dt)]
        v.index = v.index.floor("15min")
        v = v[~v.index.duplicated(keep="last")].sort_index()
        frames.append(v)
        v.reset_index(names="timestamp_utc").to_parquet(
            out_dir / "afrr_volumes_15min.parquet",
            index=False,
            compression="zstd",
        )

    combined = pd.concat(frames, axis=1, sort=False).sort_index() if frames else pd.DataFrame()
    if not combined.empty:
        combined = combined.loc[:, ~combined.columns.duplicated()]
        # Keep raw source prices untouched and add explicit pre-PICASSO
        # quarter-hour filled prices (within-hour only) for transparent VWAP use.
        cutoff = pd.Timestamp("2022-06-22 22:00:00+00:00")
        fill_pairs = [
            ("afrr_avg_activation_price_pos", "afrr_activation_price_pos_ffill"),
            ("afrr_avg_activation_price_neg", "afrr_activation_price_neg_ffill"),
        ]
        pre_mask = combined.index < cutoff
        if pre_mask.any():
            for src_col, out_col in fill_pairs:
                if src_col not in combined.columns:
                    continue
                combined[out_col] = combined[src_col]
                pre_idx = combined.index[pre_mask]
                combined.loc[pre_mask, out_col] = (
                    combined.loc[pre_mask, src_col]
                    .groupby(pre_idx.floor("1h"))
                    .ffill()
                )
        else:
            for src_col, out_col in fill_pairs:
                if src_col in combined.columns:
                    combined[out_col] = combined[src_col]
        combined.reset_index(names="timestamp_utc").to_parquet(
            out_dir / "afrr_price_volume_15min.parquet",
            index=False,
            compression="zstd",
        )
        LOGGER.info(
            "Wrote 15-minute aFRR files to %s (rows=%s).",
            out_dir,
            len(combined),
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Fetch aFRR results from Regelleistung.net",
        epilog=(
            "Example (skip bid-based activation reconstruction):\n"
            "  ./.venv/bin/python -m energy_trading.ingestion.fetch_regelleistung "
            "--start 2020-11-30T23:00:00Z --end 2025-12-31T23:00:00Z "
            "--out data/raw/regelleistung.parquet --mol-dir data/raw "
            "--skip-bid-activation-prices"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--start", required=True, help="Start ISO8601 (UTC).")
    parser.add_argument("--end", required=True, help="End ISO8601 (UTC).")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[3] / "data" / "raw" / "regelleistung.parquet"),
        help="Output parquet path",
    )
    parser.add_argument(
        "--mol-dir",
        default=str(Path(__file__).resolve().parents[3] / "data" / "raw"),
        help="Directory containing RESULT_LIST_ANONYM_* files (zip/xlsx).",
    )
    parser.add_argument(
        "--bids-dir",
        default=str(Path(__file__).resolve().parents[3] / "data" / "raw" / "bids"),
        help="Directory to store/read anonymous bid files for marginal-price calculation.",
    )
    parser.add_argument(
        "--skip-bid-activation-prices",
        action="store_true",
        help="Skip anonymous-bid activation price reconstruction (afrr_bid_* columns).",
    )
    parser.add_argument(
        "--netztransparenz-path",
        default=str(Path(__file__).resolve().parents[3] / "data" / "raw" / "netztransparenz.parquet"),
        help="Path to netztransparenz parquet used for 15-minute activated MW in VWAP calculation.",
    )
    parser.add_argument(
        "--out-15min-dir",
        default=str(Path(__file__).resolve().parents[3] / "data" / "raw" / "regelleistung_15min"),
        help="Directory for raw 15-minute aFRR price/volume parquet exports.",
    )
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    else:
        start_dt = start_dt.astimezone(timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    else:
        end_dt = end_dt.astimezone(timezone.utc)

    years = list(range(start_dt.year, end_dt.year + 1))
    all_years = []
    for y in years:
        df_cap = fetch_and_parse_regelleistung(y, "CAPACITY", start_dt=start_dt, end_dt=end_dt)
        df_ene = fetch_and_parse_regelleistung(y, "ENERGY", start_dt=start_dt, end_dt=end_dt)
        if not df_ene.empty:
            cols_15m = [
                c
                for c in (
                    "afrr_avg_activation_price_pos",
                    "afrr_avg_activation_price_neg",
                    "afrr_marginal_activation_price_pos",
                    "afrr_marginal_activation_price_neg",
                    "afrr_activation_marginal_price_pos",
                    "afrr_activation_marginal_price_neg",
                    "afrr_activation_avg_price_pos",
                    "afrr_activation_avg_price_neg",
                )
                if c in df_ene.columns
            ]
            if cols_15m:
                ALL_15MIN_PRICE_DATA.append(df_ene[cols_15m].copy())
        if not df_cap.empty or not df_ene.empty:
            all_years.append(pd.concat([df_cap, df_ene], axis=1, sort=False))

    if not all_years:
        LOGGER.warning("No regelleistung data fetched.")
        return

    df_master = pd.concat(all_years).sort_index()
    mol_df = process_mol_slope(Path(args.mol_dir))
    if not mol_df.empty:
        df_master = df_master.join(mol_df, how="left")
    if args.skip_bid_activation_prices:
        LOGGER.info("Skipping anonymous-bid activation price reconstruction (--skip-bid-activation-prices).")
    else:
        bids_dir = Path(args.bids_dir)
        download_bid_files_with_fallback(start_dt, end_dt, bids_dir)
        bid_df = process_bid_prices(bids_dir)
        if bid_df.empty and bids_dir != Path(args.mol_dir):
            LOGGER.warning(
                "No anonymous-bid prices found in %s. Falling back to %s.",
                bids_dir,
                args.mol_dir,
            )
            bid_df = process_bid_prices(Path(args.mol_dir))
        if not bid_df.empty:
            df_master = df_master.join(bid_df, how="left")
            LOGGER.info(
                "Merged anonymous-bid activation/VWAP columns into regelleistung: rows=%s, range=%s -> %s",
                len(bid_df),
                bid_df.index.min(),
                bid_df.index.max(),
            )
    df_master = df_master[~df_master.index.duplicated(keep="first")]

    # Join 15-minute activated volumes from Netztransparenz for VWAP weighting.
    netz_df = _load_netztransparenz_activation_15m(Path(args.netztransparenz_path), start_dt, end_dt)
    if not netz_df.empty:
        df_master = df_master.join(netz_df, how="left")
        LOGGER.info(
            "Joined Netztransparenz 15-minute volumes for VWAP: rows=%s, range=%s -> %s",
            len(netz_df),
            netz_df.index.min(),
            netz_df.index.max(),
        )
    else:
        LOGGER.warning(
            "No 15-minute Netztransparenz activation volumes available at %s; VWAP columns may be NaN/fallback.",
            args.netztransparenz_path,
        )

    # Persist 15-minute price and volume streams for transparent VWAP validation.
    price_15m = (
        pd.concat(ALL_15MIN_PRICE_DATA).sort_index().groupby(level=0).first()
        if ALL_15MIN_PRICE_DATA
        else pd.DataFrame()
    )
    _export_afrr_15min_price_volume(
        price_15m=price_15m,
        volume_15m=netz_df,
        out_dir=Path(args.out_15min_dir),
        start_dt=start_dt,
        end_dt=end_dt,
    )

    # Output standardization: hourly aggregation with explicit VWAP.
    df_master = _aggregate_hourly_with_vwap(df_master)
    df_master = df_master.loc[(df_master.index >= start_dt) & (df_master.index <= end_dt)]
    df_master = df_master.sort_index()
    df_master = df_master.reset_index().rename(columns={"index": "timestamp_utc"})
    if "timestamp_utc" not in df_master.columns and "timestamp" in df_master.columns:
        df_master = df_master.rename(columns={"timestamp": "timestamp_utc"})

    # Drop redundant activated volume and MWh columns (keep MW components only).
    drop_cols = [
        "mfrr_activated_mwh_pos",
        "mfrr_activated_mwh_neg",
        "afrr_activated_mwh_pos",
        "afrr_activated_mwh_neg",
        "activated_volume_pos_mwh",
        "activated_volume_neg_mwh",
        "activated_volume_pos_mw",
        "activated_volume_neg_mw",
    ]
    df_master = df_master.drop(columns=[c for c in drop_cols if c in df_master.columns])

    # Log missing months for activation market (do not impute).
    if "afrr_avg_activation_price_neg" in df_master.columns:
        missing = df_master[df_master["afrr_avg_activation_price_neg"].isna()]
        if not missing.empty:
            ts_col = "timestamp_utc" if "timestamp_utc" in missing.columns else ("timestamp" if "timestamp" in missing.columns else None)
            if ts_col is not None:
                months = (
                    pd.to_datetime(missing[ts_col], utc=True, errors="coerce")
                    .dt.strftime("%Y-%m")
                    .value_counts()
                    .sort_index()
                )
                LOGGER.warning("Missing activation market months (afrr_avg_activation_price_neg):")
                for ym, cnt in months.items():
                    LOGGER.warning("  %s: %s hours missing", ym, cnt)
            else:
                LOGGER.warning("Missing activation market months: timestamp column not found.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_master.to_parquet(out_path, index=False)
    LOGGER.info("Wrote %s rows to %s", len(df_master), out_path)


if __name__ == "__main__":
    main()
