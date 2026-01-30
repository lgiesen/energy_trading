"""Fetch Regelleistung aFRR CAPACITY/ENERGY results and store as parquet.

Usage:
    python -m ingestion.fetch_regelleistung \
        --start 2020-12-01T00:00:00Z --end 2026-01-01T02:00:00Z \
        --out data/raw/regelleistung.parquet

Outputs:
    - regelleistung.parquet with hourly aFRR capacity/energy results.

Includes:
    - capacity/energy prices and offered volumes (pos/neg)
    - net_import_export_mw (IGCC/PICASSO netting when present)
    - mol_slope_100_500 (from anonymous bid list, expanded to hourly)
"""
from __future__ import annotations

import argparse
import calendar
import io
import logging
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# Suppress openpyxl style warnings.
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
LOGGER = logging.getLogger(__name__)


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
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


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
            p100 = float(pd.Series(g[price_col]).iloc[(g["cum"] >= 100).idxmax()]) if (g["cum"] >= 100).any() else float(g[price_col].max())
            if total < 500:
                p500 = float(g[price_col].max())
            else:
                p500 = float(pd.Series(g[price_col]).iloc[(g["cum"] >= 500).idxmax()])
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
        market_type == "ENERGY"
        and start_dt is not None
        and year == start_dt.year
        and not df.empty
        and date_col is not None
    ):
        max_date = _parse_date_series(df[date_col]).max()
        if pd.notna(max_date) and max_date.date() >= start_dt.date():
            allow_partial = True
            LOGGER.info("Yearly ENERGY %s accepted as partial coverage (max date %s).", year, max_date.date())

    # If ENERGY is missing or sparse, try monthly fallbacks.
    if market_type == "ENERGY" and (df.empty or len(df) < 4000) and not allow_partial:
        LOGGER.warning(
            "Yearly ENERGY file for %s is missing or sparse (%s rows). Falling back to monthly files.",
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
                val_cols[marg_col] = "afrr_activation_price"
            if avg_col:
                val_cols[avg_col] = "afrr_activation_avg_price"
            if off_col:
                val_cols[off_col] = "afrr_activation_offered_mw"
        else:
            marg_col = _find_col(df_part, ["MARGINAL", "CAPACITY", "PRICE"]) or _find_col(df_part, ["GRENZWERT"])
            off_col = _find_col(df_part, ["OFFERED", "CAPACITY"])
            if marg_col:
                val_cols[marg_col] = "afrr_capacity_price"
            if off_col:
                val_cols[off_col] = "afrr_capacity_offered_mw"

        for col in val_cols.keys():
            df_part[col] = pd.to_numeric(df_part[col], errors="coerce").fillna(0.0)

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

        # --- 5) Pivot by direction and resample ---
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

        df_pivot = df_pivot.resample("1h").mean()
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fetch aFRR results from Regelleistung.net")
    parser.add_argument("--start", required=True, help="Start ISO8601 (UTC).")
    parser.add_argument("--end", required=True, help="End ISO8601 (UTC).")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "data" / "raw" / "regelleistung.parquet"),
        help="Output parquet path",
    )
    parser.add_argument(
        "--mol-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "raw"),
        help="Directory containing RESULT_LIST_ANONYM_* files (zip/xlsx).",
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
        if not df_cap.empty or not df_ene.empty:
            all_years.append(pd.concat([df_cap, df_ene], axis=1))

    if not all_years:
        LOGGER.warning("No regelleistung data fetched.")
        return

    df_master = pd.concat(all_years).sort_index()
    mol_df = process_mol_slope(Path(args.mol_dir))
    if not mol_df.empty:
        df_master = df_master.join(mol_df, how="left")
    df_master = df_master[~df_master.index.duplicated(keep="first")]

    # Output standardization: UTC hourly, clip window, timestamp_utc column.
    df_master = df_master.resample("1h").mean()
    df_master = df_master.loc[(df_master.index >= start_dt) & (df_master.index <= end_dt)]
    df_master = df_master.sort_index()
    df_master = df_master.reset_index().rename(columns={"index": "timestamp_utc"})
    if "timestamp_utc" not in df_master.columns and "timestamp" in df_master.columns:
        df_master = df_master.rename(columns={"timestamp": "timestamp_utc"})

    # Log missing months for activation market (do not impute).
    if "afrr_activation_avg_price_neg" in df_master.columns:
        missing = df_master[df_master["afrr_activation_avg_price_neg"].isna()]
        if not missing.empty:
            ts_col = "timestamp_utc" if "timestamp_utc" in missing.columns else ("timestamp" if "timestamp" in missing.columns else None)
            if ts_col is not None:
                months = (
                    pd.to_datetime(missing[ts_col], utc=True, errors="coerce")
                    .dt.strftime("%Y-%m")
                    .value_counts()
                    .sort_index()
                )
                LOGGER.warning("Missing activation market months (afrr_activation_avg_price_neg):")
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
