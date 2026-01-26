"""Fetch Regelleistung aFRR CAPACITY/ENERGY results and store as parquet.

Usage:
    python -m ingestion.fetch_regelleistung \
        --start-year 2022 --end-year 2025 \
        --out energy_trading/data/regelleistung.parquet

Outputs:
    - regelleistung.parquet with hourly aFRR capacity/energy results.

Includes:
    - capacity/energy prices and offered volumes (pos/neg)
    - net_import_export_mw (IGCC/PICASSO netting when present)
    - mol_slope_100_500 (from anonymous bid list, expanded to hourly)
"""
from __future__ import annotations

import argparse
import io
import warnings
import logging
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


def _mol_slope_from_bids(df_bids: pd.DataFrame) -> pd.DataFrame:
    """Compute MOL slope P500-P100 per DELIVERY_DATE + PRODUCT and expand to hourly."""
    df_bids.columns = df_bids.columns.str.strip()
    date_col = _find_col(df_bids, ["DELIVERY", "DATE"]) or _find_col(df_bids, ["DATUM"]) or df_bids.columns[0]
    prod_col = _find_col(df_bids, ["PRODUCT"]) or _find_col(df_bids, ["PRODUKT"]) or df_bids.columns[1]
    price_col = _find_col(df_bids, ["PRICE"]) or _find_col(df_bids, ["PREIS"])
    vol_col = _find_col(df_bids, ["MW"]) or _find_col(df_bids, ["CAPACITY"]) or _find_col(df_bids, ["VOLUME"])
    if not price_col or not vol_col:
        return pd.DataFrame()

    df_bids = df_bids[[date_col, prod_col, price_col, vol_col]].copy()
    df_bids[price_col] = pd.to_numeric(df_bids[price_col], errors="coerce")
    df_bids[vol_col] = pd.to_numeric(df_bids[vol_col], errors="coerce")
    df_bids = df_bids.dropna(subset=[price_col, vol_col, date_col, prod_col])

    rows = []
    for (d, prod), g in df_bids.groupby([date_col, prod_col]):
        g = g.sort_values(price_col)
        g["cum"] = g[vol_col].cumsum()
        p100 = g.loc[g["cum"] >= 100, price_col].iloc[0] if (g["cum"] >= 100).any() else g[price_col].max()
        p500 = g.loc[g["cum"] >= 500, price_col].iloc[0] if (g["cum"] >= 500).any() else g[price_col].max()
        slope = p500 - p100
        start_hour = _parse_product_start(prod)
        if start_hour is None:
            continue
        start_ts = pd.to_datetime(d) + pd.to_timedelta(start_hour, unit="h")
        for i in range(4):
            rows.append((start_ts + pd.Timedelta(hours=i), slope))

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows, columns=["timestamp", "mol_slope_100_500"])
    out["timestamp"] = (
        out["timestamp"]
        .dt.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )
    out = out.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    return out


def fetch_and_parse_regelleistung(year: int, market_type: str) -> pd.DataFrame:
    """Download aFRR results and return a tidy DataFrame indexed by timestamp."""
    url = (
        "https://www.regelleistung.net/apps/cpp-publisher/api/v2/tenders/files/"
        f"RESULT_OVERVIEW_{market_type}_MARKET_aFRR_{year}-01-01_{year}-12-31.xlsx"
    )
    LOGGER.info("Downloading %s data for %s...", market_type, year)
    try:
        resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception:
        LOGGER.warning("Skipping %s %s: file not found.", year, market_type)
        return pd.DataFrame()

    df = pd.read_excel(io.BytesIO(resp.content), engine="openpyxl")
    df.columns = df.columns.str.strip()

    # --- 1) Dynamic column detection ---
    date_col = _find_col(df, ["DATE"]) or _find_col(df, ["DATUM"]) or df.columns[0]
    prod_col = _find_col(df, ["PRODUC"]) or _find_col(df, ["PRODUKT"]) or _find_col(df, ["TIME"]) or df.columns[1]

    # --- 2) Timestamp parsing ---
    if df[prod_col].astype(str).str.contains("_").any():
        df["quarter_hour_int"] = df[prod_col].str.extract(r"_(\d+)").astype(int)
        df["time_offset"] = pd.to_timedelta((df["quarter_hour_int"] - 1) * 15, unit="m")
        df["timestamp"] = pd.to_datetime(df[date_col]) + df["time_offset"]
        df["Richtung"] = df[prod_col].str.split("_").str[0]
    else:
        df["start_time"] = df[prod_col].astype(str).str.split(" - ").str[0]
        df["timestamp"] = pd.to_datetime(df[date_col].astype(str) + " " + df["start_time"])
        dir_col = _find_col(df, ["DIRECTION"]) or _find_col(df, ["RICHTUNG"]) or df.columns[2]
        df["Richtung"] = df[dir_col]

    df["timestamp"] = (
        df["timestamp"]
        .dt.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )
    df = df.dropna(subset=["timestamp"])

    # --- 3) Feature columns ---
    val_cols: dict[str, str] = {}
    if market_type == "ENERGY":
        marg_col = _find_col(df, ["MARGINAL", "ENERGY", "PRICE"])
        avg_col = _find_col(df, ["AVERAGE", "ENERGY", "PRICE"])
        off_col = _find_col(df, ["OFFERED", "CAPACITY"])
        if marg_col:
            val_cols[marg_col] = "afrr_activation_price"
        if avg_col:
            val_cols[avg_col] = "afrr_activation_avg_price"
        if off_col:
            val_cols[off_col] = "afrr_activation_offered_mw"
    else:
        marg_col = _find_col(df, ["MARGINAL", "CAPACITY", "PRICE"]) or _find_col(df, ["GRENZWERT"])
        off_col = _find_col(df, ["OFFERED", "CAPACITY"])
        if marg_col:
            val_cols[marg_col] = "afrr_capacity_price"
        if off_col:
            val_cols[off_col] = "afrr_capacity_offered_mw"

    for col in val_cols.keys():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # --- 4) European netting (import/export) ---
    net_col = _find_col(df, ["GERMANY_IMPORT", "EXPORT"])
    net_series = None
    if net_col:
        net_series = (
            df[["timestamp", net_col]]
            .assign(**{net_col: pd.to_numeric(df[net_col], errors="coerce")})
            .groupby("timestamp")[net_col]
            .mean()
            .rename("net_import_export_mw")
        )

    # --- 5) Pivot by direction and resample ---
    df_clean = df[["timestamp", "Richtung"] + list(val_cols.keys())]
    df_pivot = df_clean.pivot_table(index="timestamp", columns="Richtung", values=list(val_cols.keys()))

    new_cols = []
    for col in df_pivot.columns:
        orig_metric = col[0]
        direction = str(col[1]).replace("ATIVE", "").lower()  # POSITIVE -> pos, NEGATIVE -> neg
        new_cols.append(f"{val_cols[orig_metric]}_{direction}")
    df_pivot.columns = new_cols

    if net_series is not None:
        df_pivot = df_pivot.join(net_series, how="left")

    df_pivot = df_pivot.resample("1h").mean()
    return df_pivot


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fetch aFRR results from Regelleistung.net")
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "data" / "regelleistung.parquet"),
        help="Output parquet path",
    )
    args = parser.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    all_years = []
    all_slopes = []
    for y in years:
        df_cap = fetch_and_parse_regelleistung(y, "CAPACITY")
        df_ene = fetch_and_parse_regelleistung(y, "ENERGY")
        try:
            url = (
                "https://www.regelleistung.net/apps/cpp-publisher/api/v2/tenders/files/"
                f"RESULT_LIST_ANONYM_ENERGY_MARKET_aFRR_{y}-01-01_{y}-12-31.xlsx"
            )
            resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            df_bids = pd.read_excel(io.BytesIO(resp.content), engine="openpyxl")
            slope_df = _mol_slope_from_bids(df_bids)
            if not slope_df.empty:
                all_slopes.append(slope_df)
        except Exception:
            LOGGER.warning("Skipping %s MOL slope: file not found or unexpected format.", y)
        if not df_cap.empty or not df_ene.empty:
            all_years.append(pd.concat([df_cap, df_ene], axis=1))

    if not all_years:
        LOGGER.warning("No regelleistung data fetched.")
        return

    df_master = pd.concat(all_years).sort_index()
    if all_slopes:
        slope_master = pd.concat(all_slopes).sort_index()
        df_master = df_master.join(slope_master, how="left")
        df_master["mol_slope_100_500"] = df_master["mol_slope_100_500"].ffill()
    df_master = df_master[~df_master.index.duplicated(keep="first")]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_master.to_parquet(out_path)
    LOGGER.info("Wrote %s rows to %s", len(df_master), out_path)


if __name__ == "__main__":
    main()
