"""
Fetch commodity prices (TTF gas, CO2, API2 coal) from Yahoo Finance and store as parquet.

Usage:
    ./.venv/bin/python -m energy_trading.ingestion.fetch_yfinance \
        --start 2020-11-30T23:00:00Z --end 2025-12-31T23:00:00Z \
        --out data/raw/yfinance.parquet

        Outputs:
    - yfinance.parquet with daily prices upsampled to hourly (ffill).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf

# Monkey-patch Timestamp.utcnow to the recommended Timestamp.now('UTC') until yfinance/pandas switch.
pd.Timestamp.utcnow = staticmethod(lambda: pd.Timestamp.now("UTC"))
LOGGER = logging.getLogger(__name__)

# Yahoo tickers (adjusted close) with fallbacks to reduce missingness:
# - TTF gas
# - API2 coal
# - CO2 EUA
CANDIDATE_TICKERS: Dict[str, list[str]] = {
    "gas_price_ttf": ["TTF=F"],
    "coal_price_api2": ["MTF=F"],
    "co2_price_eua": ["CO2.L", "CBU2.DE"],
}


def _configure_third_party_logging() -> None:
    # Keep pipeline logs readable: yfinance emits error logs for stale/delisted symbols.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def _download_adj_close(ticker: str, start: str, end: str, interval: str) -> pd.Series | None:
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )
    except Exception as exc:
        LOGGER.warning("Failed to download %s: %s", ticker, exc)
        return None
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if "Adj Close" not in df.columns:
        return None
    s = df["Adj Close"].copy()
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def fetch_prices(start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Download adjusted close prices using multiple tickers per series and coalesce."""
    series_list = []

    for col_name, tickers in CANDIDATE_TICKERS.items():
        candidates = []
        for t in tickers:
            s = _download_adj_close(t, start, end, interval)
            if s is None or s.empty:
                LOGGER.warning("No data for %s (%s)", col_name, t)
                continue
            candidates.append(s.rename(t))
        if not candidates:
            LOGGER.warning("No usable ticker for %s", col_name)
            continue
        df_cand = pd.concat(candidates, axis=1, sort=False).sort_index()
        # Coalesce across candidate tickers (first non-null).
        df_cand[col_name] = df_cand.bfill(axis=1).iloc[:, 0]
        series_list.append(df_cand[[col_name]])

    if not series_list:
        # Keep schema stable even when Yahoo has no usable symbols in this run.
        idx = pd.date_range(start=pd.to_datetime(start, utc=True), end=pd.to_datetime(end, utc=True), freq="1D")
        empty = pd.DataFrame({"timestamp": idx})
        for col_name in CANDIDATE_TICKERS:
            empty[col_name] = pd.NA
        return empty

    merged = pd.concat(series_list, axis=1, sort=False).sort_index()
    for col_name in CANDIDATE_TICKERS:
        if col_name not in merged.columns:
            merged[col_name] = pd.NA
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.reset_index()
    if "timestamp" not in merged.columns:
        if "index" in merged.columns:
            merged = merged.rename(columns={"index": "timestamp"})
        elif "Date" in merged.columns:
            merged = merged.rename(columns={"Date": "timestamp"})
    return merged


def upsample_daily_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Upsample daily closes to hourly with causal +1d availability shift.

    yfinance daily close is timestamped at day start. To avoid midnight look-ahead,
    a close observed for day D is made available from D+1 00:00 UTC onward.
    """
    if df.empty:
        return df
    if "timestamp" not in df.columns:
        return df

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    if out.empty:
        return out.reset_index()

    # Causal delay: close(D) becomes available at D+1 00:00 UTC.
    out.index = out.index + pd.Timedelta(days=1)

    start = out.index.min().floor("D")
    end = out.index.max().ceil("D")
    daily_index = pd.date_range(start, end, freq="1D", tz="UTC")
    out = out.reindex(daily_index, method="ffill")

    hourly_index = pd.date_range(start, end, freq="1h", tz="UTC")
    out = out.reindex(hourly_index, method="ffill")
    # Fill only the initial 24h gap caused by the +1d causal shift.
    out = out.bfill(limit=24)
    return out.reset_index().rename(columns={"index": "timestamp"})


def verify_yfinance_integrity(
    df_hourly: pd.DataFrame,
    df_daily_raw: pd.DataFrame | None = None,
    *,
    atol: float = 1e-9,
) -> None:
    """Verify missing-value quality and causal +1d delay semantics."""
    if df_hourly is None or df_hourly.empty:
        LOGGER.warning("Skipping yfinance integrity checks: hourly dataframe is empty.")
        return

    required_cols = ["gas_price_ttf", "coal_price_api2", "co2_price_eua"]
    missing = [c for c in required_cols if c not in df_hourly.columns]
    if missing:
        raise KeyError(f"Missing required yfinance columns: {missing}")

    # Test A: no NaNs in required commodity columns.
    for col in required_cols:
        nulls = int(df_hourly[col].isnull().sum())
        if nulls != 0:
            raise AssertionError(f"[FAIL] Test A (No NaNs): {col} has {nulls} null values")
    LOGGER.info("[PASS] Test A (No NaNs): all required yfinance columns are complete.")

    # Test B: causal delay (Date_T 08:00 equals close(Date_T-1), not close(Date_T)).
    if df_daily_raw is None or df_daily_raw.empty:
        LOGGER.warning("Skipping Test B: no daily raw reference dataframe provided.")
        return
    if "timestamp" not in df_daily_raw.columns:
        LOGGER.warning("Skipping Test B: daily raw reference has no timestamp column.")
        return

    daily = df_daily_raw.copy()
    daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True, errors="coerce").dt.floor("D")
    daily = daily.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    hourly = df_hourly.copy()
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True, errors="coerce")
    hourly = hourly.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    daily = daily.set_index("timestamp")

    for col in required_cols:
        if col not in daily.columns:
            continue
        s = pd.to_numeric(daily[col], errors="coerce")
        s_prev = s.shift(1)
        changed = (s.notna() & s_prev.notna() & ((s - s_prev).abs() > atol))
        change_days = s.index[changed]
        if len(change_days) == 0:
            LOGGER.warning("Skipping Test B for %s: no daily price-change day found.", col)
            continue

        checked = False
        for day in change_days:
            ts_8 = day + pd.Timedelta(hours=8)
            if ts_8 not in hourly.index:
                continue
            observed = pd.to_numeric(pd.Series([hourly.at[ts_8, col]]), errors="coerce").iloc[0]
            prev_close = s_prev.loc[day]
            curr_close = s.loc[day]
            if pd.isna(observed) or pd.isna(prev_close) or pd.isna(curr_close):
                continue

            if not np.isclose(observed, prev_close, atol=atol, rtol=0.0):
                raise AssertionError(
                    f"[FAIL] Test B ({col}): {ts_8} value {observed} != previous close {prev_close}"
                )
            if np.isclose(observed, curr_close, atol=atol, rtol=0.0):
                raise AssertionError(
                    f"[FAIL] Test B ({col}): {ts_8} value equals same-day close ({curr_close}); "
                    "causal delay violated"
                )
            checked = True
            LOGGER.info("[PASS] Test B (%s): causal delay verified at %s.", col, ts_8)
            break

        if not checked:
            LOGGER.warning("Skipping Test B for %s: no suitable change-day timestamp with hourly data.", col)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _configure_third_party_logging()
    parser = argparse.ArgumentParser(description="Fetch TTF gas, EUA CO2, and API2 coal prices from Yahoo Finance.")
    parser.add_argument("--start", required=True, help="Start ISO8601 (UTC).")
    parser.add_argument("--end", required=True, help="End ISO8601 (UTC).")
    parser.add_argument("--out", default="data/raw/yfinance.parquet", help="Output parquet path.")
    parser.add_argument("--interval", default="1d", help="Yahoo interval (default 1d).")
    args = parser.parse_args()

    start_dt = pd.to_datetime(args.start, utc=True)
    end_dt = pd.to_datetime(args.end, utc=True)
    # yfinance expects naive dates; use UTC dates without timezone.
    df_daily = fetch_prices(start_dt.tz_localize(None), end_dt.tz_localize(None), interval=args.interval)
    df = upsample_daily_to_hourly(df_daily)
    # Enforce UTC, hourly truncation, and clip
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("1h")
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)].sort_values("timestamp")
    verify_yfinance_integrity(df, df_daily_raw=df_daily)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="zstd", index=False)
    LOGGER.info("Wrote %s rows to %s", len(df), out_path)


if __name__ == "__main__":
    main()
