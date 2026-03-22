#!/usr/bin/env python3
"""Audit cyclical time features in final feature parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_features_path() -> Path:
    candidates = [Path("data/features/all_data_features.parquet")]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find all_data_features.parquet in expected locations.")


def main() -> None:
    path = _resolve_features_path()
    df = pd.read_parquet(path)

    required = [
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "month_sin",
        "month_cos",
        "hour",
        "dayofweek",
        "month",
    ]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing required columns: {missing}"

    # 1) Unit Circle Test (hour embedding sample)
    sample = df[["hour_sin", "hour_cos"]].dropna()
    if len(sample) > 5000:
        sample = sample.sample(n=5000, random_state=42)
    identity_err = ((sample["hour_sin"] ** 2 + sample["hour_cos"] ** 2) - 1.0).abs().max()
    assert float(identity_err) < 1e-12, f"Unit circle identity failed: max_err={identity_err}"
    print(f"[PASS] Unit Circle Test: max_abs_err={float(identity_err):.3e}")

    # 2) Boundary Jump Test: distance(23,00) == distance(12,13)
    h23 = df[df["hour"] == 23][["hour_sin", "hour_cos"]].dropna().head(1)
    h00 = df[df["hour"] == 0][["hour_sin", "hour_cos"]].dropna().head(1)
    h12 = df[df["hour"] == 12][["hour_sin", "hour_cos"]].dropna().head(1)
    h13 = df[df["hour"] == 13][["hour_sin", "hour_cos"]].dropna().head(1)
    assert len(h23) == len(h00) == len(h12) == len(h13) == 1, "Insufficient hour rows for boundary test."
    d_23_00 = np.linalg.norm(h23.iloc[0].to_numpy() - h00.iloc[0].to_numpy())
    d_12_13 = np.linalg.norm(h12.iloc[0].to_numpy() - h13.iloc[0].to_numpy())
    assert np.isclose(d_23_00, d_12_13, atol=1e-12), (
        f"Boundary jump failed: d23_00={d_23_00}, d12_13={d_12_13}"
    )
    print(f"[PASS] Boundary Jump Test: d23_00={d_23_00:.12f}, d12_13={d_12_13:.12f}")

    # 3) Range Check: all sin/cos in [-1, 1]
    cyc_cols = ["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "month_sin", "month_cos"]
    for c in cyc_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        assert s.notna().all(), f"NaNs found in {c}"
        mn, mx = float(s.min()), float(s.max())
        assert mn >= -1.0 - 1e-12 and mx <= 1.0 + 1e-12, f"Range violation in {c}: [{mn}, {mx}]"
    print("[PASS] Range Check: all cyclical columns within [-1, 1]")

    # 4) Column and NaN check (explicit)
    for c in cyc_cols:
        assert c in df.columns, f"Missing cyclical column: {c}"
        assert int(df[c].isna().sum()) == 0, f"NaNs in cyclical column: {c}"
    print("[PASS] Column Check: all 6 cyclical columns exist and contain no NaNs")

    # Thesis table for midnight adjacency demonstration
    ts = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")
    hour = float(ts.hour)
    dow = float(ts.dayofweek)
    month_raw = float(ts.month)
    month_zero = month_raw - 1.0
    table = pd.DataFrame(
        [
            {
                "timestamp_utc": str(ts),
                "hour": hour,
                "hour_sin": np.sin(2 * np.pi * hour / 24.0),
                "hour_cos": np.cos(2 * np.pi * hour / 24.0),
                "dayofweek": dow,
                "dayofweek_sin": np.sin(2 * np.pi * dow / 7.0),
                "dayofweek_cos": np.cos(2 * np.pi * dow / 7.0),
                "month_raw": month_raw,
                "month_zero_based": month_zero,
                "month_sin": np.sin(2 * np.pi * month_zero / 12.0),
                "month_cos": np.cos(2 * np.pi * month_zero / 12.0),
            }
        ]
    )
    print("\nThesis translation table (midnight adjacency proof row):")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
