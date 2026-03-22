#!/usr/bin/env python3
"""Verify outage causality in final features.

Checks that for each hour T:
    FEATURES.unplanned_outages_mw[T] == RAW.unplanned_outages_mw[T-2h]

Usage:
    ./.venv/bin/python scripts/verify_outage_causality.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _resolve_features_path(required_col: str = "unplanned_outages_mw") -> Path:
    candidates = [Path("data/features/all_data_features.parquet")]
    for path in candidates:
        if not path.exists():
            continue
        try:
            cols = set(pd.read_parquet(path).columns)
        except Exception:
            continue
        if required_col in cols:
            return path
    raise FileNotFoundError(
        f"Could not find features parquet containing `{required_col}`. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def _load_ts_df(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp_utc" not in df.columns:
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "timestamp_utc"})
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.reset_index().columns[0]: "timestamp_utc"})
    if "timestamp_utc" not in df.columns:
        raise KeyError(f"{path} missing timestamp_utc/timestamp")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    if df["timestamp_utc"].isna().any():
        raise ValueError(f"{path} contains invalid timestamps")
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def _max_abs_error(a: pd.Series, b: pd.Series) -> float:
    aa, bb = a.align(b, join="inner")
    valid = aa.notna() & bb.notna()
    if valid.sum() == 0:
        return 0.0
    diff = (pd.to_numeric(aa[valid], errors="coerce") - pd.to_numeric(bb[valid], errors="coerce")).abs()
    return float(diff.max()) if not diff.empty else 0.0


def main() -> None:
    raw_path = Path("data/processed/outages_hourly.parquet")
    feat_path = _resolve_features_path(required_col="unplanned_outages_mw")
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw outage sidecar: {raw_path}")

    raw = _load_ts_df(raw_path)
    feat = _load_ts_df(feat_path)

    for required in ("unplanned_outages_mw",):
        if required not in raw.columns:
            raise KeyError(f"RAW missing required column: {required}")
        if required not in feat.columns:
            raise KeyError(f"FEATURES missing required column: {required}")

    # Keep only the columns needed for this strict causality proof.
    raw = raw[["timestamp_utc", "unplanned_outages_mw"]].copy()
    feat = feat[["timestamp_utc", "unplanned_outages_mw"]].copy()

    joined = feat.merge(raw, on="timestamp_utc", how="left", suffixes=("_feat", "_raw"))
    joined["unplanned_outages_mw_raw"] = pd.to_numeric(joined["unplanned_outages_mw_raw"], errors="coerce").fillna(0.0)
    joined["unplanned_outages_mw_feat"] = pd.to_numeric(joined["unplanned_outages_mw_feat"], errors="coerce")

    expected = joined["unplanned_outages_mw_raw"].shift(2).fillna(0.0)
    err = _max_abs_error(joined["unplanned_outages_mw_feat"], expected)

    print(f"features_path={feat_path}")
    print(f"raw_path={raw_path}")
    print(f"max_abs_error_unplanned_lag2={err:.12g}")

    if err != 0.0:
        bad = (joined["unplanned_outages_mw_feat"] - expected).abs()
        bad = bad[bad > 0].index
        sample_idx = bad[:5].tolist()
        raise AssertionError(
            "Outage causality check failed: "
            "unplanned_outages_mw does not match RAW shift(2). "
            f"max_abs_error={err}, sample_rows={sample_idx}"
        )

    print("[PASS] Outage causality verified: unplanned_outages_mw(T) == RAW(T-2h)")


if __name__ == "__main__":
    main()
