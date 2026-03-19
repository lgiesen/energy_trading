"""Refine merged market data with domain-specific consolidation and features.

Usage:
    ./.venv/bin/python -m energy_trading.processing.refine_market_data \
        --in data/processed/all_data.parquet \
        --out data/processed/all_data_refined.parquet
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import polars as pl

LOGGER = logging.getLogger(__name__)
PICASSO_START_UTC = datetime(2022, 6, 22, 22, 0, tzinfo=timezone.utc)
DEFAULT_REGELLEISTUNG_15M_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "raw"
    / "regelleistung_15min"
    / "afrr_price_volume_15min.parquet"
)
DEFAULT_BIDS_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "bids"

SMARD_REDUNDANT_COLS = [
    "wind_onshore_actual",
    "wind_offshore_actual",
    "solar_actual",
    "wind_onshore_forecast",
    "wind_offshore_forecast",
    "solar_forecast",
    "wind_onshore_error",
    "wind_offshore_error",
    "solar_error",
]


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _median_abs(df: pl.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    val = df.select(pl.col(col).cast(pl.Float64, strict=False).abs().median()).item()
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _find_col(columns: list[str], needles: list[str]) -> str | None:
    for c in columns:
        c_up = str(c).upper()
        if all(n.upper() in c_up for n in needles):
            return c
    return None


def _parse_date_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")
    serial = pd.to_numeric(series, errors="coerce")
    use_serial = serial.notna() & (parsed.isna() | (parsed.dt.year < 1995))
    if use_serial.any():
        parsed.loc[use_serial] = pd.to_datetime(
            serial.loc[use_serial], unit="D", origin="1899-12-30", errors="coerce"
        )
    return parsed


def _local_berlin_to_utc(ts: pd.Series) -> tuple[pd.Series, str]:
    """Convert local German time series to UTC with DST-safe handling."""
    ts = pd.to_datetime(ts, errors="coerce")
    tz_status = "unknown"
    if getattr(ts.dt, "tz", None) is None:
        tz_status = "naive_localized_berlin_to_utc"
        ts = ts.dt.tz_localize("Europe/Berlin", ambiguous="infer", nonexistent="shift_forward")
    else:
        tzname = str(ts.dt.tz)
        if "UTC" in tzname.upper():
            tz_status = "already_utc"
        else:
            tz_status = f"aware_{tzname}_to_utc"
    ts = ts.dt.tz_convert("UTC")
    return ts, tz_status


def _iter_bid_files(bids_dir: Path) -> list[Path]:
    if not bids_dir.exists():
        return []
    files = [
        p
        for p in bids_dir.rglob("RESULT_LIST_ANONYM_ENERGY_MARKET_*.xlsx*")
        if p.is_file() and not p.name.startswith("~$")
    ]
    # Prefer zipped sources when both zipped and plain xlsx exist.
    selected: dict[str, Path] = {}
    for p in files:
        key = p.name.replace(".zip", "")
        if key not in selected or p.suffix.lower() == ".zip":
            selected[key] = p
    return sorted(selected.values())


def _read_bid_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith((".xlsx", ".xls")):
                    with zf.open(name) as f:
                        return pd.read_excel(f, engine="openpyxl")
        return pd.DataFrame()
    return pd.read_excel(path, engine="openpyxl")


def _parse_bid_features_15m(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame()
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)

    date_col = _find_col(cols, ["DELIVERY", "DATE"]) or _find_col(cols, ["DATUM"])
    prod_col = _find_col(cols, ["PRODUCT"]) or _find_col(cols, ["PRODUKT"])
    alloc_col = _find_col(cols, ["ALLOCATED", "CAPACITY"]) or _find_col(cols, ["CAPACITY", "MW"])
    price_col = _find_col(cols, ["ENERGY", "PRICE"]) or _find_col(cols, ["PRICE"])
    pay_col = _find_col(cols, ["PAYMENT", "DIRECTION"]) or _find_col(cols, ["ENERGY_PRICE_PAYMENT_DIRECTION"])
    reserve_col = _find_col(cols, ["RESERVE"]) or _find_col(cols, ["RESERVETYPE"])
    country_col = _find_col(cols, ["COUNTRY"])

    if not date_col or not prod_col or not alloc_col:
        return pd.DataFrame()

    keep = [date_col, prod_col, alloc_col]
    if price_col:
        keep.append(price_col)
    if pay_col:
        keep.append(pay_col)
    if reserve_col:
        keep.append(reserve_col)
    if country_col:
        keep.append(country_col)
    df = df[keep].copy()

    df[date_col] = _parse_date_series(df[date_col])
    df[alloc_col] = pd.to_numeric(df[alloc_col], errors="coerce")
    if price_col:
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[date_col, alloc_col])
    df = df[df[alloc_col] > 0]

    if reserve_col:
        df = df[df[reserve_col].astype(str).str.upper().str.contains("AFRR", na=False)]
    if country_col:
        df = df[df[country_col].astype(str).str.upper().eq("DE")]

    prod = df[prod_col].astype(str)
    df["direction"] = prod.str.split("_").str[0].str.upper()
    df = df[df["direction"].isin(["POS", "NEG"])]

    qh = pd.to_numeric(prod.str.extract(r"_(\d{3})$")[0], errors="coerce")
    sh = pd.to_numeric(prod.str.extract(r"_(\d{2})_(\d{2})$")[0], errors="coerce")
    eh = pd.to_numeric(prod.str.extract(r"_(\d{2})_(\d{2})$")[1], errors="coerce")
    mask_qh = qh.notna()
    mask_block = sh.notna() & eh.notna()

    if price_col and pay_col:
        pay = df[pay_col].astype(str).str.upper()
        price_sign = np.where(pay.str.contains("PROVIDER_TO_GRID", na=False), -1.0, 1.0)
        df["signed_energy_price_eur_mwh"] = df[price_col] * price_sign
        df["is_provider_to_grid"] = pay.str.contains("PROVIDER_TO_GRID", na=False).astype(float)
    else:
        df["signed_energy_price_eur_mwh"] = np.nan
        df["is_provider_to_grid"] = np.nan

    parts: list[pd.DataFrame] = []
    if mask_qh.any():
        q = df.loc[mask_qh].copy()
        idx = qh.loc[mask_qh].astype(int)
        base_local = pd.to_datetime(q[date_col]).dt.floor("D").dt.tz_localize(
            "Europe/Berlin", ambiguous="NaT", nonexistent="shift_forward"
        )
        q_ts = (base_local + pd.to_timedelta((idx - 1) * 15, unit="m")).dt.tz_convert("UTC")
        q["timestamp_utc"] = q_ts
        parts.append(
            q[
                [
                    "timestamp_utc",
                    "direction",
                    alloc_col,
                    "signed_energy_price_eur_mwh",
                    "is_provider_to_grid",
                ]
            ]
        )

    if mask_block.any():
        b = df.loc[mask_block].copy()
        start_h = sh.loc[mask_block].astype(int).to_numpy()
        end_h = eh.loc[mask_block].astype(int).to_numpy()
        base_local = pd.to_datetime(b[date_col]).dt.floor("D").dt.tz_localize(
            "Europe/Berlin", ambiguous="NaT", nonexistent="shift_forward"
        )
        rows: list[tuple[pd.Timestamp, str, float, float, float]] = []
        for ts0, d, alloc, price_signed, p2g, s_h, e_h in zip(
            base_local,
            b["direction"],
            b[alloc_col],
            b["signed_energy_price_eur_mwh"],
            b["is_provider_to_grid"],
            start_h,
            end_h,
        ):
            if pd.isna(ts0):
                continue
            end_adj = e_h if e_h > s_h else e_h + 24
            periods = max(0, (end_adj - s_h) * 4)
            if periods == 0:
                continue
            block = pd.date_range(
                start=ts0 + pd.to_timedelta(s_h, unit="h"),
                periods=periods,
                freq="15min",
                inclusive="left",
            )
            for ts in block:
                rows.append((ts.tz_convert("UTC"), d, float(alloc), float(price_signed), float(p2g)))
        if rows:
            parts.append(
                pd.DataFrame(
                    rows,
                    columns=[
                        "timestamp_utc",
                        "direction",
                        alloc_col,
                        "signed_energy_price_eur_mwh",
                        "is_provider_to_grid",
                    ],
                )
            )

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp_utc"])
    out["price_weighted"] = out["signed_energy_price_eur_mwh"] * out[alloc_col]
    agg = (
        out.groupby(["timestamp_utc", "direction"], as_index=False)
        .agg(
            total_awarded_capacity_mw=(alloc_col, "sum"),
            sum_price_weighted=("price_weighted", "sum"),
            provider_to_grid_share=("is_provider_to_grid", "mean"),
        )
    )
    agg["bid_signed_vwap_eur_mwh"] = np.where(
        agg["total_awarded_capacity_mw"] != 0,
        agg["sum_price_weighted"] / agg["total_awarded_capacity_mw"],
        np.nan,
    )

    cap_piv = agg.pivot_table(
        index="timestamp_utc",
        columns="direction",
        values="total_awarded_capacity_mw",
        aggfunc="sum",
    )
    cap_piv.columns = [f"awarded_capacity_mw_{str(c).lower()}" for c in cap_piv.columns]
    cap_piv = cap_piv.reset_index()

    vwap_piv = agg.pivot_table(
        index="timestamp_utc",
        columns="direction",
        values="bid_signed_vwap_eur_mwh",
        aggfunc="mean",
    )
    vwap_piv.columns = [f"bid_signed_vwap_eur_mwh_{str(c).lower()}" for c in vwap_piv.columns]
    vwap_piv = vwap_piv.reset_index()

    p2g_piv = agg.pivot_table(
        index="timestamp_utc",
        columns="direction",
        values="provider_to_grid_share",
        aggfunc="mean",
    )
    p2g_piv.columns = [f"bid_provider_to_grid_share_{str(c).lower()}" for c in p2g_piv.columns]
    p2g_piv = p2g_piv.reset_index()

    piv = cap_piv.merge(vwap_piv, on="timestamp_utc", how="outer").merge(p2g_piv, on="timestamp_utc", how="outer")
    for c in ("awarded_capacity_mw_pos", "awarded_capacity_mw_neg"):
        if c not in piv.columns:
            piv[c] = np.nan
    for c in (
        "bid_signed_vwap_eur_mwh_pos",
        "bid_signed_vwap_eur_mwh_neg",
        "bid_provider_to_grid_share_pos",
        "bid_provider_to_grid_share_neg",
    ):
        if c not in piv.columns:
            piv[c] = np.nan
    return piv[
        [
            "timestamp_utc",
            "awarded_capacity_mw_pos",
            "awarded_capacity_mw_neg",
            "bid_signed_vwap_eur_mwh_pos",
            "bid_signed_vwap_eur_mwh_neg",
            "bid_provider_to_grid_share_pos",
            "bid_provider_to_grid_share_neg",
        ]
    ]


def load_bid_hourly_features_from_bids(bids_dir: Path) -> pl.DataFrame:
    cache_path = bids_dir / "_afrr_bid_hourly_cache.parquet"
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            if {"timestamp", "bid_alloc_mw_pos", "bid_alloc_mw_neg"}.issubset(set(cached.columns)):
                ts_utc, tz_status = _local_berlin_to_utc(cached["timestamp"])
                cached["timestamp_utc"] = ts_utc
                cached["awarded_capacity_mw_pos"] = pd.to_numeric(cached["bid_alloc_mw_pos"], errors="coerce")
                cached["awarded_capacity_mw_neg"] = pd.to_numeric(cached["bid_alloc_mw_neg"], errors="coerce")
                if "afrr_bid_vwap_activation_price_pos" in cached.columns:
                    cached["bid_signed_vwap_eur_mwh_pos"] = pd.to_numeric(
                        cached["afrr_bid_vwap_activation_price_pos"], errors="coerce"
                    )
                else:
                    cached["bid_signed_vwap_eur_mwh_pos"] = np.nan
                if "afrr_bid_vwap_activation_price_neg" in cached.columns:
                    cached["bid_signed_vwap_eur_mwh_neg"] = pd.to_numeric(
                        cached["afrr_bid_vwap_activation_price_neg"], errors="coerce"
                    )
                else:
                    cached["bid_signed_vwap_eur_mwh_neg"] = np.nan
                cached["bid_provider_to_grid_share_pos"] = np.nan
                cached["bid_provider_to_grid_share_neg"] = np.nan
                cached = cached.dropna(subset=["timestamp_utc"])
                cached = (
                    cached.groupby("timestamp_utc", as_index=False)[
                        [
                            "awarded_capacity_mw_pos",
                            "awarded_capacity_mw_neg",
                            "bid_signed_vwap_eur_mwh_pos",
                            "bid_signed_vwap_eur_mwh_neg",
                            "bid_provider_to_grid_share_pos",
                            "bid_provider_to_grid_share_neg",
                        ]
                    ]
                    .mean()
                    .sort_values("timestamp_utc")
                )
                out = pl.from_pandas(cached).with_columns(
                    pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False)
                ).sort("timestamp_utc")
                LOGGER.info("Loaded awarded capacity from bid cache: %s rows", out.height)
                LOGGER.info("Bid cache timezone conversion: %s", tz_status)
                return out
        except Exception as exc:
            LOGGER.warning("Failed reading bid cache %s: %s", cache_path, exc)

    files = _iter_bid_files(bids_dir)
    if not files:
        LOGGER.warning("No anonymous bid files found in %s for awarded-capacity aggregation.", bids_dir)
        return pl.DataFrame()

    parts: list[pd.DataFrame] = []
    for path in files:
        try:
            raw = _read_bid_df(path)
        except Exception as exc:
            LOGGER.warning("Skip bid file %s: %s", path.name, exc)
            continue
        parsed = _parse_bid_features_15m(raw)
        if not parsed.empty:
            parts.append(parsed)

    if not parts:
        LOGGER.warning("No awarded-capacity rows parsed from %s.", bids_dir)
        return pl.DataFrame()

    df_15m = pd.concat(parts, ignore_index=True)
    df_15m["timestamp_utc"] = pd.to_datetime(df_15m["timestamp_utc"], utc=True, errors="coerce")
    df_15m = df_15m.dropna(subset=["timestamp_utc"])
    df_15m = (
        df_15m.groupby("timestamp_utc", as_index=False)
        .agg(
            awarded_capacity_mw_pos=("awarded_capacity_mw_pos", "sum"),
            awarded_capacity_mw_neg=("awarded_capacity_mw_neg", "sum"),
            bid_signed_vwap_eur_mwh_pos=("bid_signed_vwap_eur_mwh_pos", "mean"),
            bid_signed_vwap_eur_mwh_neg=("bid_signed_vwap_eur_mwh_neg", "mean"),
            bid_provider_to_grid_share_pos=("bid_provider_to_grid_share_pos", "mean"),
            bid_provider_to_grid_share_neg=("bid_provider_to_grid_share_neg", "mean"),
        )
        .sort_values("timestamp_utc")
    )
    # Align denominator resolution with hourly activated MW in the main dataset.
    hourly = (
        df_15m.set_index("timestamp_utc")
        .resample("1h")
        .mean(numeric_only=True)
        .reset_index()
    )
    out = pl.from_pandas(hourly)
    return out.with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False)
    ).sort("timestamp_utc")


def _parse_bid_merit_rows_15m(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Parse anonymous bid rows into 15-minute UTC merit-order points."""
    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)

    date_col = _find_col(cols, ["DELIVERY", "DATE"]) or _find_col(cols, ["DATUM"])
    prod_col = _find_col(cols, ["PRODUCT"]) or _find_col(cols, ["PRODUKT"])
    offered_col = _find_col(cols, ["OFFERED", "CAPACITY"]) or _find_col(cols, ["OFFERED", "MW"])
    price_col = _find_col(cols, ["ENERGY", "PRICE"]) or _find_col(cols, ["PRICE"])
    pay_col = _find_col(cols, ["PAYMENT", "DIRECTION"]) or _find_col(cols, ["ENERGY_PRICE_PAYMENT_DIRECTION"])
    reserve_col = _find_col(cols, ["RESERVE"]) or _find_col(cols, ["RESERVETYPE"])
    country_col = _find_col(cols, ["COUNTRY"])

    if not date_col or not prod_col or not offered_col or not price_col:
        return pd.DataFrame()

    keep = [date_col, prod_col, offered_col, price_col]
    if pay_col:
        keep.append(pay_col)
    if reserve_col:
        keep.append(reserve_col)
    if country_col:
        keep.append(country_col)
    df = df[keep].copy()

    df[date_col] = _parse_date_series(df[date_col])
    df[offered_col] = pd.to_numeric(df[offered_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[date_col, offered_col, price_col])
    df = df[df[offered_col] > 0]

    if reserve_col:
        df = df[df[reserve_col].astype(str).str.upper().str.contains("AFRR", na=False)]
    if country_col:
        df = df[df[country_col].astype(str).str.upper().eq("DE")]

    prod = df[prod_col].astype(str)
    df["direction"] = prod.str.split("_").str[0].str.upper()
    df = df[df["direction"].isin(["POS", "NEG"])]

    pay = df[pay_col].astype(str).str.upper() if pay_col else pd.Series("", index=df.index)
    price_sign = np.where(pay.str.contains("PROVIDER_TO_GRID", na=False), -1.0, 1.0)
    df["payment_direction"] = np.where(
        pay.str.contains("PROVIDER_TO_GRID", na=False),
        "PROVIDER_TO_GRID",
        "GRID_TO_PROVIDER",
    )
    df["energy_price_eur_mwh"] = df[price_col]
    df["signed_energy_price_eur_mwh"] = df["energy_price_eur_mwh"] * price_sign

    qh = pd.to_numeric(prod.str.extract(r"_(\d{3})$")[0], errors="coerce")
    sh = pd.to_numeric(prod.str.extract(r"_(\d{2})_(\d{2})$")[0], errors="coerce")
    eh = pd.to_numeric(prod.str.extract(r"_(\d{2})_(\d{2})$")[1], errors="coerce")
    mask_qh = qh.notna()
    mask_block = sh.notna() & eh.notna()

    parts: list[pd.DataFrame] = []
    if mask_qh.any():
        q = df.loc[mask_qh].copy()
        idx = qh.loc[mask_qh].astype(int)
        base_local = pd.to_datetime(q[date_col]).dt.floor("D").dt.tz_localize(
            "Europe/Berlin", ambiguous="NaT", nonexistent="shift_forward"
        )
        q["timestamp_utc"] = (base_local + pd.to_timedelta((idx - 1) * 15, unit="m")).dt.tz_convert("UTC")
        parts.append(
            q[
                [
                    "timestamp_utc",
                    "direction",
                    "payment_direction",
                    "energy_price_eur_mwh",
                    "signed_energy_price_eur_mwh",
                    offered_col,
                ]
            ]
        )

    if mask_block.any():
        b = df.loc[mask_block].copy()
        start_h = sh.loc[mask_block].astype(int).to_numpy()
        end_h = eh.loc[mask_block].astype(int).to_numpy()
        base_local = pd.to_datetime(b[date_col]).dt.floor("D").dt.tz_localize(
            "Europe/Berlin", ambiguous="NaT", nonexistent="shift_forward"
        )
        rows: list[tuple[pd.Timestamp, str, str, float, float, float]] = []
        for ts0, d, pay_dir, price_raw, price_signed, offered, s_h, e_h in zip(
            base_local,
            b["direction"],
            b["payment_direction"],
            b["energy_price_eur_mwh"],
            b["signed_energy_price_eur_mwh"],
            b[offered_col],
            start_h,
            end_h,
        ):
            if pd.isna(ts0):
                continue
            end_adj = e_h if e_h > s_h else e_h + 24
            periods = max(0, (end_adj - s_h) * 4)
            if periods == 0:
                continue
            block = pd.date_range(
                start=ts0 + pd.to_timedelta(s_h, unit="h"),
                periods=periods,
                freq="15min",
                inclusive="left",
            )
            for ts in block:
                rows.append(
                    (
                        ts.tz_convert("UTC"),
                        d,
                        str(pay_dir),
                        float(price_raw),
                        float(price_signed),
                        float(offered),
                    )
                )
        if rows:
            parts.append(
                pd.DataFrame(
                    rows,
                    columns=[
                        "timestamp_utc",
                        "direction",
                        "payment_direction",
                        "energy_price_eur_mwh",
                        "signed_energy_price_eur_mwh",
                        offered_col,
                    ],
                )
            )

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp_utc"])
    out = out.rename(columns={offered_col: "offered_capacity_mw"})
    out["offered_capacity_mw"] = pd.to_numeric(out["offered_capacity_mw"], errors="coerce")
    out["energy_price_eur_mwh"] = pd.to_numeric(out["energy_price_eur_mwh"], errors="coerce")
    out["signed_energy_price_eur_mwh"] = pd.to_numeric(out["signed_energy_price_eur_mwh"], errors="coerce")
    out = out.dropna(subset=["offered_capacity_mw", "energy_price_eur_mwh", "signed_energy_price_eur_mwh"])
    out = out[out["offered_capacity_mw"] > 0]
    return out[
        [
            "timestamp_utc",
            "direction",
            "payment_direction",
            "offered_capacity_mw",
            "energy_price_eur_mwh",
            "signed_energy_price_eur_mwh",
        ]
    ]


def _pick_activation_series_15m(df_15m: pl.DataFrame, direction: str) -> tuple[str | None, bool]:
    candidates = (
        [f"afrr_activated_mw_{direction}", f"abgerufene_arbeit_{direction}", f"afrr_activated_mwh_{direction}"]
        if direction == "pos"
        else [f"afrr_activated_mw_{direction}", f"abgerufene_arbeit_{direction}", f"afrr_activated_mwh_{direction}"]
    )
    col = _first_existing(df_15m.columns, candidates)
    if col is None:
        return None, False
    col_l = col.lower()
    needs_x4 = "mwh" in col_l
    return col, needs_x4


def _reconstruct_marginal_for_direction(
    bids_15m: pl.DataFrame,
    activation_15m: pl.DataFrame,
    direction: str,
    out_col: str,
) -> pl.DataFrame:
    if bids_15m.is_empty() or activation_15m.is_empty():
        return pl.DataFrame(schema={"timestamp_utc": pl.Datetime(time_unit="us", time_zone="UTC"), out_col: pl.Float64})

    b = bids_15m.filter(pl.col("direction") == direction)
    if b.is_empty():
        return pl.DataFrame(schema={"timestamp_utc": pl.Datetime(time_unit="us", time_zone="UTC"), out_col: pl.Float64})

    if direction == "POS":
        b = b.with_columns(
            [
                pl.lit(0).alias("__neg_rank"),
                pl.col("energy_price_eur_mwh").alias("__neg_sort_key"),
            ]
        )
    else:
        b = b.with_columns(
            [
                pl.when(pl.col("payment_direction") == "PROVIDER_TO_GRID")
                .then(pl.lit(0))
                .otherwise(pl.lit(1))
                .alias("__neg_rank"),
                pl.when(pl.col("payment_direction") == "PROVIDER_TO_GRID")
                .then(-pl.col("energy_price_eur_mwh"))
                .otherwise(pl.col("energy_price_eur_mwh"))
                .alias("__neg_sort_key"),
            ]
        )

    b = (
        b.sort(["timestamp_utc", "__neg_rank", "__neg_sort_key"])
        .with_columns(
            pl.col("offered_capacity_mw")
            .cast(pl.Float64, strict=False)
            .cum_sum()
            .over("timestamp_utc")
            .alias("__cum_offered_mw")
        )
    )

    a = (
        activation_15m
        .select(["timestamp_utc", "activation_mw"])
        .with_columns(pl.col("activation_mw").cast(pl.Float64, strict=False).alias("activation_mw"))
    )
    a_pos = a.filter(pl.col("activation_mw") > 0).sort(["timestamp_utc", "activation_mw"])
    if a_pos.is_empty():
        return pl.DataFrame(schema={"timestamp_utc": pl.Datetime(time_unit="us", time_zone="UTC"), out_col: pl.Float64})

    curve = b.select(["timestamp_utc", "__cum_offered_mw", "signed_energy_price_eur_mwh"]).sort(
        ["timestamp_utc", "__cum_offered_mw"]
    )

    last_price = (
        curve.group_by("timestamp_utc")
        .agg(pl.col("signed_energy_price_eur_mwh").last().alias("__last_price"))
    )

    # Merit-order match without grouped join_asof warning:
    # for each timestamp, pick the first bid where cum_offered >= activation.
    hit = (
        a_pos.join(curve, on="timestamp_utc", how="left")
        .filter(pl.col("__cum_offered_mw") >= pl.col("activation_mw"))
        .group_by(["timestamp_utc", "activation_mw"])
        .agg(
            pl.col("signed_energy_price_eur_mwh")
            .sort_by("__cum_offered_mw")
            .first()
            .alias("__hit_price")
        )
    )
    matched = (
        a_pos.join(hit, on=["timestamp_utc", "activation_mw"], how="left")
        .join(last_price, on="timestamp_utc", how="left")
        .with_columns(
            pl.coalesce(
                [
                    pl.col("__hit_price").cast(pl.Float64, strict=False),
                    pl.col("__last_price").cast(pl.Float64, strict=False),
                ]
            ).alias(out_col)
        )
        .select(["timestamp_utc", out_col])
    )

    zeros = (
        a.filter(pl.col("activation_mw") <= 0)
        .select(["timestamp_utc"])
        .with_columns(pl.lit(float("nan")).alias(out_col))
    )

    return pl.concat([matched, zeros], how="diagonal_relaxed").sort("timestamp_utc")


def compute_afrr_reconstructed_marginal_from_bids(
    bids_dir: Path, regelleistung_15m_path: Path
) -> pl.DataFrame:
    """Reconstruct 15-minute local-German marginal activation price from bids."""
    if not regelleistung_15m_path.exists():
        LOGGER.warning("Skip simulated marginal: missing 15-minute regelleistung source %s", regelleistung_15m_path)
        return pl.DataFrame()

    files = _iter_bid_files(bids_dir)
    if not files:
        LOGGER.warning("Skip simulated marginal: no anonymous bid files in %s", bids_dir)
        return pl.DataFrame()

    raw_15m = pl.read_parquet(regelleistung_15m_path).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False)
    )
    pos_col, pos_x4 = _pick_activation_series_15m(raw_15m, "pos")
    neg_col, neg_x4 = _pick_activation_series_15m(raw_15m, "neg")
    if pos_col is None and neg_col is None:
        LOGGER.warning("Skip simulated marginal: no activation columns found in %s", regelleistung_15m_path)
        return pl.DataFrame()

    act_exprs: list[pl.Expr] = [pl.col("timestamp_utc")]
    if pos_col is not None:
        pos_expr = pl.col(pos_col).cast(pl.Float64, strict=False)
        if pos_x4:
            pos_expr = pos_expr * 4.0
        act_exprs.append(pos_expr.abs().alias("__act_pos_mw"))
        LOGGER.info("Simulated marginal POS activation source: %s%s", pos_col, " (*4)" if pos_x4 else "")
    if neg_col is not None:
        neg_expr = pl.col(neg_col).cast(pl.Float64, strict=False)
        if neg_x4:
            neg_expr = neg_expr * 4.0
        act_exprs.append(neg_expr.abs().alias("__act_neg_mw"))
        LOGGER.info("Simulated marginal NEG activation source: %s%s", neg_col, " (*4)" if neg_x4 else "")

    act = raw_15m.select(act_exprs).drop_nulls(["timestamp_utc"]).sort("timestamp_utc").to_pandas()
    act["timestamp_utc"] = pd.to_datetime(act["timestamp_utc"], utc=True, errors="coerce")
    act = act.dropna(subset=["timestamp_utc"])
    act = act[act["timestamp_utc"] >= pd.Timestamp(PICASSO_START_UTC)]

    cache_path = bids_dir / "_afrr_bid_merit_15m_cache.parquet"
    if cache_path.exists():
        try:
            merit = pd.read_parquet(cache_path)
            # Backward compatibility: older cache versions may miss newly
            # required fields. Reconstruct them so we can reuse cache and avoid
            # expensive workbook parsing.
            if "energy_price_eur_mwh" not in merit.columns and "signed_energy_price_eur_mwh" in merit.columns:
                merit["energy_price_eur_mwh"] = pd.to_numeric(
                    merit["signed_energy_price_eur_mwh"], errors="coerce"
                ).abs()
            if "payment_direction" not in merit.columns:
                if {"signed_energy_price_eur_mwh", "direction", "energy_price_eur_mwh"}.issubset(set(merit.columns)):
                    signed = pd.to_numeric(merit["signed_energy_price_eur_mwh"], errors="coerce")
                    raw = pd.to_numeric(merit["energy_price_eur_mwh"], errors="coerce")
                    dir_sign = np.where(merit["direction"].astype(str).str.upper().eq("NEG"), -1.0, 1.0)
                    # signed = raw * pay_sign * dir_sign  -> pay_sign = signed/(raw*dir_sign)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        pay_sign = signed / (raw * dir_sign)
                    merit["payment_direction"] = np.where(
                        pay_sign < 0,
                        "PROVIDER_TO_GRID",
                        "GRID_TO_PROVIDER",
                    )
                else:
                    merit["payment_direction"] = "GRID_TO_PROVIDER"
            merit["timestamp_utc"] = pd.to_datetime(merit["timestamp_utc"], utc=True, errors="coerce")
            merit = merit.dropna(
                subset=[
                    "timestamp_utc",
                    "offered_capacity_mw",
                    "energy_price_eur_mwh",
                    "signed_energy_price_eur_mwh",
                    "payment_direction",
                ]
            )
            merit = merit[merit["timestamp_utc"] >= pd.Timestamp(PICASSO_START_UTC)]
            # Persist migrated cache schema so future runs stay fast and quiet.
            try:
                merit.to_parquet(cache_path, index=False)
            except Exception:
                pass
            LOGGER.info("Loaded bid merit cache: %s rows", len(merit))
        except Exception as exc:
            LOGGER.warning("Failed reading bid merit cache %s: %s", cache_path, exc)
            merit = pd.DataFrame()
    else:
        merit = pd.DataFrame()

    if merit.empty:
        merit_parts: list[pd.DataFrame] = []
        for path in files:
            year_match = pd.Series([path.name]).str.extract(r"(20\d{2})")[0].iloc[0]
            if year_match is not None:
                try:
                    if int(year_match) < 2022:
                        continue
                except Exception:
                    pass
            try:
                raw = _read_bid_df(path)
            except Exception as exc:
                LOGGER.warning("Skip bid file %s for simulated marginal: %s", path.name, exc)
                continue
            parsed = _parse_bid_merit_rows_15m(raw)
            if not parsed.empty:
                parsed = parsed[parsed["timestamp_utc"] >= pd.Timestamp(PICASSO_START_UTC)]
                if not parsed.empty:
                    merit_parts.append(parsed)

        if not merit_parts:
            LOGGER.warning("Skip simulated marginal: no parseable merit-order rows in %s", bids_dir)
            return pl.DataFrame()

        merit = pd.concat(merit_parts, ignore_index=True)
        merit["timestamp_utc"] = pd.to_datetime(merit["timestamp_utc"], utc=True, errors="coerce")
        merit = merit.dropna(
            subset=[
                "timestamp_utc",
                "offered_capacity_mw",
                "energy_price_eur_mwh",
                "signed_energy_price_eur_mwh",
                "payment_direction",
            ]
        )
        merit = merit[merit["timestamp_utc"] >= pd.Timestamp(PICASSO_START_UTC)]
        try:
            merit.to_parquet(cache_path, index=False)
            LOGGER.info("Wrote bid merit cache: %s rows -> %s", len(merit), cache_path)
        except Exception as exc:
            LOGGER.warning("Could not write bid merit cache %s: %s", cache_path, exc)

    merit = merit.sort_values(["timestamp_utc", "direction", "offered_capacity_mw"])
    merit_pl = pl.from_pandas(merit).with_columns(
        [
            pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False),
            pl.col("offered_capacity_mw").cast(pl.Float64, strict=False),
            pl.col("energy_price_eur_mwh").cast(pl.Float64, strict=False),
            pl.col("signed_energy_price_eur_mwh").cast(pl.Float64, strict=False),
            pl.col("payment_direction").cast(pl.Utf8, strict=False),
            pl.col("direction").cast(pl.Utf8, strict=False),
        ]
    )
    act_pl = pl.from_pandas(act).with_columns(
        [pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False)]
    )
    act_pos = (
        act_pl.select(["timestamp_utc", pl.col("__act_pos_mw").cast(pl.Float64, strict=False).alias("activation_mw")])
        if "__act_pos_mw" in act_pl.columns
        else pl.DataFrame()
    )
    act_neg = (
        act_pl.select(["timestamp_utc", pl.col("__act_neg_mw").cast(pl.Float64, strict=False).abs().alias("activation_mw")])
        if "__act_neg_mw" in act_pl.columns
        else pl.DataFrame()
    )

    pos_rec = _reconstruct_marginal_for_direction(
        merit_pl, act_pos, "POS", "afrr_reconstructed_marginal_price_pos"
    )
    neg_rec = _reconstruct_marginal_for_direction(
        merit_pl, act_neg, "NEG", "afrr_reconstructed_marginal_price_neg"
    )
    merged = (
        act_pl.select(["timestamp_utc"]).unique().sort("timestamp_utc")
        .join(pos_rec, on="timestamp_utc", how="left")
        .join(neg_rec, on="timestamp_utc", how="left")
    )

    # Validation sample: 5 high-activation rows (POS) vs official average (if available in 15m source).
    official_col = _first_existing(raw_15m.columns, ["afrr_avg_activation_price_pos", "durchschnittlicher_arbeitspreis_pos"])
    act_pd = act_pl.to_pandas()
    merged_pd = merged.to_pandas()
    if "__act_pos_mw" in act_pd.columns:
        sample = merged_pd.merge(act_pd[["timestamp_utc", "__act_pos_mw"]], on="timestamp_utc", how="left")
        if official_col is not None:
            official = raw_15m.select(
                [
                    pl.col("timestamp_utc"),
                    pl.col(official_col).cast(pl.Float64, strict=False).alias("__official_avg_price_pos"),
                ]
            ).to_pandas()
            official["timestamp_utc"] = pd.to_datetime(official["timestamp_utc"], utc=True, errors="coerce")
            sample = sample.merge(official, on="timestamp_utc", how="left")
        sample = sample.sort_values("__act_pos_mw", ascending=False).head(5)
        if not sample.empty:
            cols_show = ["timestamp_utc", "__act_pos_mw", "afrr_reconstructed_marginal_price_pos"]
            if "__official_avg_price_pos" in sample.columns:
                cols_show.append("__official_avg_price_pos")
            LOGGER.info(
                "Simulated marginal validation sample (top-5 POS activation, local German merit-order):\n%s",
                sample[cols_show].to_string(index=False),
            )

    return merged.with_columns(
        [
            pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False),
            pl.lit(True).alias("is_local_reconstruction_only"),
        ]
    ).sort("timestamp_utc")


def compute_afrr_vwap_from_15min(path: Path) -> pl.DataFrame:
    """Compute hourly aFRR VWAP from 15-minute Regelleistung price+volume data.

    Method:
    - Use activated work/volume as weight.
    - Pre-PICASSO (< 2022-06-22 22:00 UTC): forward-fill hourly posted prices
      inside each hour so :15/:30/:45 can be weighted correctly.
    - Aggregate with group_by_dynamic(every="1h", closed="left").
    - If hourly volume sum is 0 or null, return NaN VWAP.
    """
    if not path.exists():
        LOGGER.info("No 15-minute Regelleistung file found for VWAP: %s", path)
        return pl.DataFrame()

    df = pl.read_parquet(path)
    if "timestamp_utc" not in df.columns:
        LOGGER.warning("Skip VWAP build: missing timestamp_utc in %s", path)
        return pl.DataFrame()

    price_pos_col = _first_existing(
        df.columns,
        [
            "afrr_activation_price_pos_ffill",
            "afrr_avg_activation_price_pos",
            "afrr_activation_avg_price_pos",
            "afrr_marginal_activation_price_pos",
            "afrr_activation_marginal_price_pos",
            "arbeitspreis_pos",  # legacy alias in older exports
        ],
    )
    price_neg_col = _first_existing(
        df.columns,
        [
            "afrr_activation_price_neg_ffill",
            "afrr_avg_activation_price_neg",
            "afrr_activation_avg_price_neg",
            "afrr_marginal_activation_price_neg",
            "afrr_activation_marginal_price_neg",
            "arbeitspreis_neg",  # legacy alias in older exports
        ],
    )
    vol_pos_col = _first_existing(
        df.columns,
        ["afrr_activated_mw_pos", "activated_volume_pos_mw", "abgerufene_arbeit_pos"],
    )
    vol_neg_col = _first_existing(
        df.columns,
        ["afrr_activated_mw_neg", "activated_volume_neg_mw", "abgerufene_arbeit_neg"],
    )

    has_pos = price_pos_col is not None and vol_pos_col is not None
    has_neg = price_neg_col is not None and vol_neg_col is not None
    if not has_pos and not has_neg:
        LOGGER.warning("Skip VWAP build: no usable 15-minute aFRR price/volume columns in %s", path)
        return pl.DataFrame()

    cast_exprs = [
        pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False),
    ]
    for c in (price_pos_col, price_neg_col, vol_pos_col, vol_neg_col):
        if c is not None:
            cast_exprs.append(pl.col(c).cast(pl.Float64, strict=False))
    df = (
        df.with_columns(cast_exprs)
        .drop_nulls(subset=["timestamp_utc"])
        .sort("timestamp_utc")
    )

    cutoff = pl.lit(PICASSO_START_UTC).cast(pl.Datetime(time_unit="us", time_zone="UTC"))
    ff_exprs: list[pl.Expr] = []
    if has_pos:
        if price_pos_col == "afrr_activation_price_pos_ffill":
            ff_exprs.append(pl.col(price_pos_col).alias("__price_pos_ff"))
        else:
            ff_exprs.append(
                pl.when(pl.col("timestamp_utc") < cutoff)
                .then(pl.col(price_pos_col).forward_fill().over(pl.col("timestamp_utc").dt.truncate("1h")))
                .otherwise(pl.col(price_pos_col))
                .alias("__price_pos_ff")
            )
    if has_neg:
        if price_neg_col == "afrr_activation_price_neg_ffill":
            ff_exprs.append(pl.col(price_neg_col).alias("__price_neg_ff"))
        else:
            ff_exprs.append(
                pl.when(pl.col("timestamp_utc") < cutoff)
                .then(pl.col(price_neg_col).forward_fill().over(pl.col("timestamp_utc").dt.truncate("1h")))
                .otherwise(pl.col(price_neg_col))
                .alias("__price_neg_ff")
            )
    df = df.with_columns(ff_exprs)

    calc_exprs: list[pl.Expr] = []
    agg_exprs: list[pl.Expr] = []
    if has_pos:
        calc_exprs.append((pl.col("__price_pos_ff") * pl.col(vol_pos_col)).alias("__weighted_cost_pos"))
        agg_exprs.extend(
            [
                pl.col("__weighted_cost_pos").sum().alias("__sum_weighted_pos"),
                pl.col(vol_pos_col).sum().alias("__sum_vol_pos"),
            ]
        )
    if has_neg:
        calc_exprs.append((pl.col("__price_neg_ff") * pl.col(vol_neg_col)).alias("__weighted_cost_neg"))
        agg_exprs.extend(
            [
                pl.col("__weighted_cost_neg").sum().alias("__sum_weighted_neg"),
                pl.col(vol_neg_col).sum().alias("__sum_vol_neg"),
            ]
        )

    hourly = (
        df.with_columns(calc_exprs)
        .group_by_dynamic(
            index_column="timestamp_utc",
            every="1h",
            period="1h",
            closed="left",
            label="left",
        )
        .agg(agg_exprs)
        .sort("timestamp_utc")
    )

    vwap_exprs: list[pl.Expr] = []
    out_cols: list[str] = []
    if has_pos:
        out_cols.append("afrr_vwap_pos")
        vwap_exprs.append(
            pl.when(pl.col("__sum_vol_pos").is_null() | (pl.col("__sum_vol_pos") == 0.0))
            .then(pl.lit(float("nan")))
            .otherwise(pl.col("__sum_weighted_pos") / pl.col("__sum_vol_pos"))
            .alias("afrr_vwap_pos")
        )
    if has_neg:
        out_cols.append("afrr_vwap_neg")
        vwap_exprs.append(
            pl.when(pl.col("__sum_vol_neg").is_null() | (pl.col("__sum_vol_neg") == 0.0))
            .then(pl.lit(float("nan")))
            .otherwise(pl.col("__sum_weighted_neg") / pl.col("__sum_vol_neg"))
            .alias("afrr_vwap_neg")
        )

    return hourly.with_columns(vwap_exprs).select(["timestamp_utc"] + out_cols)


def _add_if_possible(
    exprs: list[pl.Expr],
    created: list[str],
    cols: list[str],
    out_col: str,
    expr: pl.Expr,
    available_cols: list[str],
) -> None:
    if all(c in available_cols for c in cols):
        exprs.append(expr.alias(out_col))
        created.append(out_col)
    else:
        missing = [c for c in cols if c not in available_cols]
        LOGGER.warning("Skip %s, missing inputs: %s", out_col, missing)


def add_specialized_activation_rates(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add deterministic physical/ML activation-rate columns on hourly data.

    Definitions (hourly):
    - activation_rate_phys_* = activated_mw / capacity_awarded_mw
      signed and uncapped to preserve physical stress/scarcity behavior.
    - activation_rate_ml_* = min(|activated_mw| / capacity_awarded_mw, 1.0)
      absolute and capped for stable ML targets.

    Division-by-zero policy:
    - if awarded capacity <= 0, set rate to 0.0.

    ACHTUNG: Kein Faktor 4 nötig, da Daten stündlich aggregiert sind
    (1 MW * 1h = 1 MWh).
    """
    schema_cols = set(lf.collect_schema().names())
    required = {
        "afrr_activated_mw_pos",
        "afrr_activated_mw_neg",
        "afrr_capacity_awarded_mw_pos",
        "afrr_capacity_awarded_mw_neg",
    }
    missing = sorted(required - schema_cols)
    if missing:
        LOGGER.warning("Skip specialized activation rates, missing inputs: %s", missing)
        return lf

    num_pos = pl.col("afrr_activated_mw_pos").cast(pl.Float64, strict=False)
    num_neg = pl.col("afrr_activated_mw_neg").cast(pl.Float64, strict=False)
    den_pos = pl.col("afrr_capacity_awarded_mw_pos").cast(pl.Float64, strict=False)
    den_neg = pl.col("afrr_capacity_awarded_mw_neg").cast(pl.Float64, strict=False)

    phys_pos_expr = (
        pl.when(den_pos > 0.0)
        .then((num_pos / den_pos).cast(pl.Float64))
        .otherwise(pl.lit(0.0, dtype=pl.Float64))
    )
    phys_neg_expr = (
        pl.when(den_neg > 0.0)
        .then((num_neg / den_neg).cast(pl.Float64))
        .otherwise(pl.lit(0.0, dtype=pl.Float64))
    )
    ml_pos_expr = (
        pl.when(den_pos > 0.0)
        .then((num_pos.abs() / den_pos).clip(upper_bound=1.0).cast(pl.Float64))
        .otherwise(pl.lit(0.0, dtype=pl.Float64))
    )
    ml_neg_expr = (
        pl.when(den_neg > 0.0)
        .then((num_neg.abs() / den_neg).clip(upper_bound=1.0).cast(pl.Float64))
        .otherwise(pl.lit(0.0, dtype=pl.Float64))
    )

    return lf.with_columns(
        [
            phys_pos_expr.alias("activation_rate_phys_pos"),
            phys_neg_expr.alias("activation_rate_phys_neg"),
            ml_pos_expr.alias("activation_rate_ml_pos"),
            ml_neg_expr.alias("activation_rate_ml_neg"),
            # Backward-compatible aliases for existing downstream usage.
            phys_pos_expr.alias("afrr_activation_rate_pos"),
            phys_neg_expr.alias("afrr_activation_rate_neg"),
        ]
    )


def refine(df: pl.DataFrame) -> pl.DataFrame:
    if "timestamp_utc" not in df.columns:
        raise ValueError("Missing required column: timestamp_utc")

    # Build and join hourly VWAP from raw 15-minute Regelleistung exports.
    vwap_hourly = compute_afrr_vwap_from_15min(DEFAULT_REGELLEISTUNG_15M_PATH)
    if not vwap_hourly.is_empty():
        df = df.join(vwap_hourly, on="timestamp_utc", how="left", suffix="_from15m")
        for col in ("afrr_vwap_pos", "afrr_vwap_neg"):
            from15 = f"{col}_from15m"
            if from15 in df.columns and col in df.columns:
                df = df.with_columns(pl.coalesce([pl.col(from15), pl.col(col)]).alias(col)).drop(from15)
            elif from15 in df.columns:
                df = df.rename({from15: col})
        LOGGER.info("Joined hourly aFRR VWAP from 15-minute source: %s rows", vwap_hourly.height)
    else:
        LOGGER.info("No 15-minute VWAP source joined; continuing with existing hourly columns.")

    # Reconstruct 15-minute local marginal activation price from anonymous bids
    # and align to hourly dataset via hourly mean of quarter-hour marginals.
    marginal_15m = compute_afrr_reconstructed_marginal_from_bids(
        DEFAULT_BIDS_DIR, DEFAULT_REGELLEISTUNG_15M_PATH
    )
    if not marginal_15m.is_empty():
        marginal_hourly = (
            marginal_15m
            .group_by_dynamic(
                index_column="timestamp_utc",
                every="1h",
                period="1h",
                closed="left",
                label="left",
            )
            .agg(
                [
                    pl.col("afrr_reconstructed_marginal_price_pos").mean(),
                    pl.col("afrr_reconstructed_marginal_price_neg").mean(),
                    pl.col("is_local_reconstruction_only").max().alias("is_local_reconstruction_only"),
                ]
            )
            .sort("timestamp_utc")
        )
        df = df.join(marginal_hourly, on="timestamp_utc", how="left", suffix="_from15m")
        for col in (
            "afrr_reconstructed_marginal_price_pos",
            "afrr_reconstructed_marginal_price_neg",
            "is_local_reconstruction_only",
        ):
            from15 = f"{col}_from15m"
            if from15 in df.columns and col in df.columns:
                df = df.with_columns(pl.coalesce([pl.col(from15), pl.col(col)]).alias(col)).drop(from15)
            elif from15 in df.columns:
                df = df.rename({from15: col})
        LOGGER.info(
            "Joined hourly simulated marginal prices from 15-minute merit-order reconstruction: %s rows",
            marginal_hourly.height,
        )
    else:
        LOGGER.info("No simulated marginal source joined from bids.")

    cutoff_bool = pl.lit(PICASSO_START_UTC).cast(pl.Datetime(time_unit="us", time_zone="UTC"))
    if "is_local_reconstruction_only" in df.columns:
        df = df.with_columns(
            pl.coalesce(
                [
                    pl.col("is_local_reconstruction_only").cast(pl.Boolean, strict=False),
                    (pl.col("timestamp_utc") >= cutoff_bool),
                ]
            ).alias("is_local_reconstruction_only")
        )
    else:
        df = df.with_columns((pl.col("timestamp_utc") >= cutoff_bool).alias("is_local_reconstruction_only"))

    # Aggregate awarded capacity from anonymous bids and join.
    bid_hourly = load_bid_hourly_features_from_bids(DEFAULT_BIDS_DIR)
    if not bid_hourly.is_empty():
        df = df.join(bid_hourly, on="timestamp_utc", how="left", suffix="_from_bids")
        for c in (
            "awarded_capacity_mw_pos",
            "awarded_capacity_mw_neg",
            "bid_signed_vwap_eur_mwh_pos",
            "bid_signed_vwap_eur_mwh_neg",
            "bid_provider_to_grid_share_pos",
            "bid_provider_to_grid_share_neg",
        ):
            c_b = f"{c}_from_bids"
            if c_b in df.columns and c in df.columns:
                df = df.with_columns(pl.coalesce([pl.col(c_b), pl.col(c)]).alias(c)).drop(c_b)
            elif c_b in df.columns:
                df = df.rename({c_b: c})
        LOGGER.info("Joined hourly bid features from anonymous bids: %s rows", bid_hourly.height)
    else:
        LOGGER.warning("No bid-derived hourly features joined; activation-rate denominator may be unavailable.")

    # Canonical awarded-capacity aliases for downstream audits.
    if "awarded_capacity_mw_pos" in df.columns and "afrr_capacity_awarded_mw_pos" not in df.columns:
        df = df.with_columns(pl.col("awarded_capacity_mw_pos").alias("afrr_capacity_awarded_mw_pos"))
    if "awarded_capacity_mw_neg" in df.columns and "afrr_capacity_awarded_mw_neg" not in df.columns:
        df = df.with_columns(pl.col("awarded_capacity_mw_neg").alias("afrr_capacity_awarded_mw_neg"))

    # Signed bid-VWAP override (economic sign from ENERGY_PRICE_PAYMENT_DIRECTION).
    if "bid_signed_vwap_eur_mwh_pos" in df.columns:
        if "afrr_vwap_pos" in df.columns:
            df = df.with_columns(
                pl.coalesce([pl.col("bid_signed_vwap_eur_mwh_pos"), pl.col("afrr_vwap_pos")]).alias(
                    "afrr_vwap_pos"
                )
            )
        else:
            df = df.with_columns(pl.col("bid_signed_vwap_eur_mwh_pos").alias("afrr_vwap_pos"))
    if "bid_signed_vwap_eur_mwh_neg" in df.columns:
        if "afrr_vwap_neg" in df.columns:
            df = df.with_columns(
                pl.coalesce([pl.col("bid_signed_vwap_eur_mwh_neg"), pl.col("afrr_vwap_neg")]).alias(
                    "afrr_vwap_neg"
                )
            )
        else:
            df = df.with_columns(pl.col("bid_signed_vwap_eur_mwh_neg").alias("afrr_vwap_neg"))

    # Normalize VWAP dtype.
    for c in ("afrr_vwap_pos", "afrr_vwap_neg"):
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False).alias(c))

    drop_cols = [c for c in SMARD_REDUNDANT_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)
    LOGGER.info("Dropped %s redundant SMARD columns: %s", len(drop_cols), drop_cols)

    exprs: list[pl.Expr] = []
    created: list[str] = []
    cols = df.columns

    _add_if_possible(
        exprs,
        created,
        [
            "wind_onshore_forecast_da_entsoe",
            "wind_offshore_forecast_da_entsoe",
            "wind_onshore_actual_entsoe",
            "wind_offshore_actual_entsoe",
        ],
        "wind_total_error_da",
        (
            pl.col("wind_onshore_forecast_da_entsoe")
            + pl.col("wind_offshore_forecast_da_entsoe")
            - pl.col("wind_onshore_actual_entsoe")
            - pl.col("wind_offshore_actual_entsoe")
        ),
        cols,
    )
    _add_if_possible(
        exprs,
        created,
        ["solar_forecast_da_entsoe", "solar_actual_entsoe"],
        "solar_error_da",
        pl.col("solar_forecast_da_entsoe") - pl.col("solar_actual_entsoe"),
        cols,
    )
    _add_if_possible(
        exprs,
        created,
        ["wind_onshore_forecast_id_entsoe", "wind_onshore_forecast_da_entsoe"],
        "wind_forecast_update",
        pl.col("wind_onshore_forecast_id_entsoe") - pl.col("wind_onshore_forecast_da_entsoe"),
        cols,
    )
    # Prefer ENTSO-E load, fallback to generic load_actual when present.
    load_col = "load_actual_entsoe" if "load_actual_entsoe" in cols else ("load_actual" if "load_actual" in cols else None)
    if load_col is not None:
        _add_if_possible(
            exprs,
            created,
            [
                load_col,
                "wind_onshore_actual_entsoe",
                "wind_offshore_actual_entsoe",
                "solar_actual_entsoe",
            ],
            "residual_load_calc",
            (
                pl.col(load_col)
                - pl.col("wind_onshore_actual_entsoe")
                - pl.col("wind_offshore_actual_entsoe")
                - pl.col("solar_actual_entsoe")
            ),
            cols,
        )
    else:
        LOGGER.warning("Skip residual_load_calc, missing inputs: ['load_actual_entsoe' or 'load_actual']")
    _add_if_possible(
        exprs,
        created,
        ["afrr_picasso_mw_pos", "afrr_picasso_mw_neg"],
        "afrr_picasso_net_mw",
        pl.col("afrr_picasso_mw_pos") - pl.col("afrr_picasso_mw_neg"),
        cols,
    )
    _add_if_possible(
        exprs,
        created,
        ["afrr_picasso_mw_pos", "afrr_picasso_mw_neg"],
        "afrr_picasso_churn_mw",
        pl.col("afrr_picasso_mw_pos") + pl.col("afrr_picasso_mw_neg"),
        cols,
    )
    _add_if_possible(
        exprs,
        created,
        ["mfrr_mari_mw_pos", "mfrr_mari_mw_neg"],
        "mfrr_mari_net_mw",
        pl.col("mfrr_mari_mw_pos") - pl.col("mfrr_mari_mw_neg"),
        cols,
    )

    # Draft profitability feature:
    # Profit = (Allocated_Capacity * Capacity_Price) + (Activated_MWh * Signed_VWAP)
    # with signed VWAP already reflecting payment direction.
    act_mwh_pos = (
        pl.col("afrr_activated_mwh_pos").cast(pl.Float64, strict=False)
        if "afrr_activated_mwh_pos" in df.columns
        else (
            pl.col("afrr_activated_mw_pos").cast(pl.Float64, strict=False) * 0.25
            if "afrr_activated_mw_pos" in df.columns
            else None
        )
    )
    act_mwh_neg = (
        pl.col("afrr_activated_mwh_neg").cast(pl.Float64, strict=False)
        if "afrr_activated_mwh_neg" in df.columns
        else (
            pl.col("afrr_activated_mw_neg").cast(pl.Float64, strict=False) * 0.25
            if "afrr_activated_mw_neg" in df.columns
            else None
        )
    )
    if (
        "afrr_capacity_awarded_mw_pos" in df.columns
        and "afrr_capacity_awarded_mw_neg" in df.columns
        and "afrr_capacity_price_pos" in df.columns
        and "afrr_capacity_price_neg" in df.columns
        and "afrr_vwap_pos" in df.columns
        and "afrr_vwap_neg" in df.columns
        and act_mwh_pos is not None
        and act_mwh_neg is not None
    ):
        exprs.append(
            (
                pl.col("afrr_capacity_awarded_mw_pos").cast(pl.Float64, strict=False)
                * pl.col("afrr_capacity_price_pos").cast(pl.Float64, strict=False)
                + act_mwh_pos * pl.col("afrr_vwap_pos").cast(pl.Float64, strict=False)
                + pl.col("afrr_capacity_awarded_mw_neg").cast(pl.Float64, strict=False)
                * pl.col("afrr_capacity_price_neg").cast(pl.Float64, strict=False)
                + act_mwh_neg * pl.col("afrr_vwap_neg").cast(pl.Float64, strict=False)
            ).alias("simulated_afrr_profit_eur")
        )
        created.append("simulated_afrr_profit_eur")
    else:
        LOGGER.warning("Skip simulated_afrr_profit_eur, missing required inputs.")

    if exprs:
        df = df.with_columns(exprs)

    # Specialized deterministic activation rates (hourly; no factor-4 scaling).
    prev_cols = set(df.columns)
    df = add_specialized_activation_rates(df.lazy()).collect()
    new_rate_cols = [
        c
        for c in (
            "activation_rate_phys_pos",
            "activation_rate_phys_neg",
            "activation_rate_ml_pos",
            "activation_rate_ml_neg",
            "afrr_activation_rate_pos",
            "afrr_activation_rate_neg",
        )
        if c in df.columns and (c not in prev_cols or c.startswith("afrr_activation_rate_"))
    ]
    if new_rate_cols:
        created.extend(new_rate_cols)

    # Optional sanity check: ML rates must remain in [0, 1].
    if "activation_rate_ml_pos" in df.columns and "activation_rate_ml_neg" in df.columns:
        ml_violations = df.filter(
            (pl.col("activation_rate_ml_pos") < 0.0)
            | (pl.col("activation_rate_ml_pos") > 1.0)
            | (pl.col("activation_rate_ml_neg") < 0.0)
            | (pl.col("activation_rate_ml_neg") > 1.0)
        ).height
        if ml_violations > 0:
            LOGGER.warning("ML activation-rate range check failed: %s rows outside [0,1]", ml_violations)
        else:
            LOGGER.info("ML activation-rate range check passed: all rows in [0,1].")

    LOGGER.info("Created/updated %s columns: %s", len(created), created)

    # Structural break for platform flows in refined layer.
    cutoff = pl.lit(PICASSO_START_UTC).cast(pl.Datetime(time_unit="us", time_zone="UTC"))
    flow_cols = [
        c
        for c in (
            "afrr_picasso_mw_pos",
            "afrr_picasso_mw_neg",
            "afrr_picasso_net_mw",
            "afrr_picasso_churn_mw",
            "mfrr_mari_mw_pos",
            "mfrr_mari_mw_neg",
            "mfrr_mari_net_mw",
        )
        if c in df.columns
    ]
    if flow_cols:
        df = df.with_columns(
            [
                pl.when(pl.col("timestamp_utc") < cutoff)
                .then(pl.lit(0.0))
                .otherwise(pl.col(c))
                .cast(pl.Float64, strict=False)
                .alias(c)
                for c in flow_cols
            ]
        )

    # Keep only net flow for MARI after deriving net.
    mfrr_drop = [c for c in ("mfrr_mari_mw_pos", "mfrr_mari_mw_neg") if c in df.columns]
    if mfrr_drop:
        df = df.drop(mfrr_drop)
    LOGGER.info("Dropped %s redundant MARI columns: %s", len(mfrr_drop), mfrr_drop)

    # Preserve raw provenance columns by design.
    preserved = [c for c in df.columns if c.endswith("_qs") or c.endswith("_op") or c.endswith("source") or c.endswith("is_fallback")]
    LOGGER.info("Preserved provenance columns: %s", preserved)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Refine merged market data for downstream modeling.")
    parser.add_argument("--in", dest="input_path", default="data/processed/all_data.parquet", help="Input parquet path.")
    parser.add_argument(
        "--out",
        dest="output_path",
        default="data/processed/all_data_refined.parquet",
        help="Output refined parquet path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input parquet: {input_path}")

    df = pl.read_parquet(input_path).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False)
    ).sort("timestamp_utc")
    LOGGER.info("Loaded %s rows and %s columns from %s", df.height, len(df.columns), input_path)

    refined = refine(df)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    refined.write_parquet(output_path, compression="zstd")
    LOGGER.info("Wrote %s rows and %s columns to %s", refined.height, len(refined.columns), output_path)


if __name__ == "__main__":
    main()
