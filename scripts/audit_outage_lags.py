#!/usr/bin/env python3
"""Audit outage lag integration for causal safety.

Loads:
- RAW sidecar: data/processed/outages_hourly.parquet
- FEATURES:    data/features/all_data_features.parquet (fallback: data/processed)

Tests:
- A: unplanned_outages_mw == RAW.shift(2)
- B: planned_outages_mw   == RAW.shift(0)
- C: no NaNs in outage columns in FEATURES
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_paths() -> tuple[Path, Path]:
    raw = Path("data/processed/outages_hourly.parquet")
    features_candidates = [
        Path("data/features/all_data_features.parquet"),
        Path("data/processed/all_data_features.parquet"),
    ]
    feat = next((p for p in features_candidates if p.exists()), None)
    if not raw.exists():
        raise FileNotFoundError(f"Missing RAW outages sidecar: {raw}")
    if feat is None:
        raise FileNotFoundError(
            "Missing FEATURES parquet. Checked: "
            + ", ".join(str(p) for p in features_candidates)
        )
    # Prefer the first candidate that already contains outage columns.
    needed = {"planned_outages_mw", "unplanned_outages_mw"}
    for cand in features_candidates:
        if not cand.exists():
            continue
        cols = set(pd.read_parquet(cand).columns)
        if needed.issubset(cols):
            return raw, cand
    return raw, feat


def _load_with_timestamp(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp_utc" not in df.columns:
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "timestamp_utc"})
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            idx_name = df.columns[0]
            if idx_name != "timestamp_utc":
                df = df.rename(columns={idx_name: "timestamp_utc"})
    if "timestamp_utc" not in df.columns:
        raise KeyError(f"{path} has no timestamp_utc/timestamp column")

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
    raw_path, feat_path = _resolve_paths()
    raw = _load_with_timestamp(raw_path)
    feat = _load_with_timestamp(feat_path)

    needed_raw = {"planned_outages_mw", "unplanned_outages_mw"}
    needed_feat = {"planned_outages_mw", "unplanned_outages_mw"}
    miss_raw = sorted(list(needed_raw - set(raw.columns)))
    miss_feat = sorted(list(needed_feat - set(feat.columns)))
    if miss_raw:
        raise KeyError(f"RAW missing columns: {miss_raw}")
    if miss_feat:
        raise KeyError(f"FEATURES missing columns: {miss_feat}")

    raw = raw[["timestamp_utc", "planned_outages_mw", "unplanned_outages_mw"]].copy()
    feat = feat[["timestamp_utc", "planned_outages_mw", "unplanned_outages_mw"]].copy()

    joined = feat.merge(raw, on="timestamp_utc", how="left", suffixes=("_feat", "_raw"))

    # 0-fill for non-outage periods, per audit spec.
    for c in ("planned_outages_mw_raw", "unplanned_outages_mw_raw"):
        joined[c] = pd.to_numeric(joined[c], errors="coerce").fillna(0.0)
    for c in ("planned_outages_mw_feat", "unplanned_outages_mw_feat"):
        joined[c] = pd.to_numeric(joined[c], errors="coerce")

    # Test A: unplanned lag = 2h
    a_expected = joined["unplanned_outages_mw_raw"].shift(2).fillna(0.0)
    a_err = _max_abs_error(joined["unplanned_outages_mw_feat"], a_expected)
    a_ok = (a_err == 0.0)
    print(f"[{'PASS' if a_ok else 'FAIL'}] Test A (Unplanned Lag=2h) max_abs_error={a_err:.12g}")

    # Test B: planned lag = 0h
    b_expected = joined["planned_outages_mw_raw"].shift(0)
    b_err = _max_abs_error(joined["planned_outages_mw_feat"], b_expected)
    b_ok = (b_err == 0.0)
    print(f"[{'PASS' if b_ok else 'FAIL'}] Test B (Planned Lag=0h) max_abs_error={b_err:.12g}")

    # Test C: no NaNs in FEATURES outage columns
    c_nulls_planned = int(joined["planned_outages_mw_feat"].isna().sum())
    c_nulls_unplanned = int(joined["unplanned_outages_mw_feat"].isna().sum())
    c_ok = (c_nulls_planned == 0 and c_nulls_unplanned == 0)
    print(
        f"[{'PASS' if c_ok else 'FAIL'}] Test C (Null Leakage) "
        f"planned_nulls={c_nulls_planned}, unplanned_nulls={c_nulls_unplanned}"
    )

    # Hard assertions requested.
    assert a_ok, f"Test A failed: max_abs_error={a_err}"
    assert b_ok, f"Test B failed: max_abs_error={b_err}"
    assert c_ok, (
        "Test C failed: outage columns contain NaNs "
        f"(planned={c_nulls_planned}, unplanned={c_nulls_unplanned})"
    )

    print("All outage lag audits PASSED.")


if __name__ == "__main__":
    main()
