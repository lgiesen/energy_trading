#!/usr/bin/env python3
"""Generate a thesis-ready null report for feature artifacts.

Output schema:
- col
- null_count
- first_null_ts
- last_null_ts
- reason_category
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def reason_category(col: str) -> str:
    c = col.lower()
    if c.startswith("target_"):
        return "target_shift_boundary"
    if "co2_price" in c:
        return "source_history_gap"
    if "lag_168h" in c or "lag_48h" in c or "lag_24h" in c or "lag_12h" in c or "lag_6h" in c or "lag_4h" in c or "lag_3h" in c or "lag_2h" in c or "lag_1h" in c:
        return "lag_warmup_or_sparse_source"
    if "forecast" in c:
        return "forecast_source_gap"
    if "actual" in c or "activated" in c or "offered" in c:
        return "operational_source_gap"
    return "other_or_unknown"


def build_report(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp_utc" in df.columns:
        ts = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    else:
        ts = pd.Series(pd.NaT, index=df.index)

    rows: list[dict] = []
    for col in df.columns:
        mask = df[col].isna()
        n = int(mask.sum())
        if n == 0:
            continue
        idx = mask[mask].index
        first_ts = ts.loc[idx[0]] if len(idx) else pd.NaT
        last_ts = ts.loc[idx[-1]] if len(idx) else pd.NaT
        rows.append(
            {
                "col": col,
                "null_count": n,
                "first_null_ts": first_ts,
                "last_null_ts": last_ts,
                "reason_category": reason_category(col),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["col", "null_count", "first_null_ts", "last_null_ts", "reason_category"])
    out = out.sort_values(["null_count", "col"], ascending=[False, True]).reset_index(drop=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Generate null transparency report for feature artifact.")
    p.add_argument("--path", default="data/features/all_data_features.parquet", help="Input parquet path")
    p.add_argument("--out", default="data/reports/null_report_features.csv", help="Output CSV path")
    args = p.parse_args()

    in_path = Path(args.path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = build_report(in_path)
    report.to_csv(out_path, index=False)

    print(f"[INFO] Input: {in_path.resolve()}")
    print(f"[INFO] Columns with NaNs: {len(report)}")
    print(f"[INFO] Report written: {out_path.resolve()}")


if __name__ == "__main__":
    main()
