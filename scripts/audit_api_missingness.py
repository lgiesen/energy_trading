#!/usr/bin/env python3
"""Audit missing values across raw API source parquet files.

Outputs:
- data/reports/api_missingness_audit.csv
- docs/api_missingness_report.md
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


API_FILE_MAP = {
    "entsoe.parquet": "ENTSO-E",
    "smard.parquet": "SMARD",
    "regelleistung.parquet": "Regelleistung.net",
    "netztransparenz.parquet": "Netztransparenz",
    "energy_charts.parquet": "Energy-Charts",
    "yfinance.parquet": "Yahoo Finance",
    "april_refetch_temp.parquet": "ENTSO-E (April Re-fetch Temp)",
}


def _resolve_repo_root() -> Path:
    root = Path.cwd().resolve()
    if (root / "src").exists():
        return root
    for parent in root.parents:
        if (parent / "src").exists():
            return parent
    raise RuntimeError("Could not resolve REPO_ROOT (directory containing 'src').")


REPO_ROOT = _resolve_repo_root()


def _detect_timestamp_col(df: pd.DataFrame) -> str | None:
    preferred = [
        "timestamp_utc",
        "timestamp",
        "datetime_utc",
        "datetime",
        "date_utc",
        "date",
    ]
    for col in preferred:
        if col in df.columns:
            return col
    for col in df.columns:
        lc = col.lower()
        if "timestamp" in lc or "date" in lc or "time" in lc:
            return col
    return None


def _to_utc_series(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, utc=True, errors="coerce")
    return out


def _scan_file(path: Path, api_name: str) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    ts_col = _detect_timestamp_col(df)
    ts = _to_utc_series(df[ts_col]) if ts_col else pd.Series([pd.NaT] * len(df))

    rows = []
    for col in df.columns:
        if col == ts_col:
            continue
        s = df[col]
        null_mask = s.isna()
        null_count = int(null_mask.sum())
        if null_count == 0:
            continue
        idx = np.flatnonzero(null_mask.to_numpy())
        first_null = ts.iloc[idx[0]] if len(idx) and ts_col else pd.NaT
        last_null = ts.iloc[idx[-1]] if len(idx) and ts_col else pd.NaT
        rows.append(
            {
                "api": api_name,
                "file": str(path.relative_to(REPO_ROOT)),
                "timestamp_col": ts_col or "",
                "column": col,
                "rows": int(len(df)),
                "null_count": null_count,
                "null_pct": round((null_count / len(df)) * 100.0, 4) if len(df) else 0.0,
                "first_null_ts": first_null,
                "last_null_ts": last_null,
            }
        )
    return pd.DataFrame(rows)


def _write_md(report: pd.DataFrame, out_md: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    lines: list[str] = []
    lines.append("# API Missingness Report")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{now}`")
    lines.append(f"- Scope: `data/raw/*.parquet`")
    lines.append("")

    if report.empty:
        lines.append("Keine Missing Values in den gescannten Raw-Parquet-Dateien gefunden.")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    overview = (
        report.groupby(["api", "file"], as_index=False)
        .agg(
            columns_with_nulls=("column", "nunique"),
            total_null_cells=("null_count", "sum"),
            max_null_pct=("null_pct", "max"),
        )
        .sort_values(["total_null_cells", "columns_with_nulls"], ascending=[False, False])
    )
    lines.append("## Überblick je Quelle")
    lines.append("")
    lines.append("| API | Datei | Spalten mit NaNs | Null-Zellen gesamt | Max Null-% in einer Spalte |")
    lines.append("|---|---|---:|---:|---:|")
    for _, r in overview.iterrows():
        lines.append(
            f"| {r['api']} | `{r['file']}` | {int(r['columns_with_nulls'])} | "
            f"{int(r['total_null_cells'])} | {float(r['max_null_pct']):.2f} |"
        )
    lines.append("")

    lines.append("## Top 50 problematische Spalten")
    lines.append("")
    lines.append("| API | Spalte | Null-% | Null Count | First Null TS | Last Null TS |")
    lines.append("|---|---|---:|---:|---|---|")
    top = report.sort_values(["null_pct", "null_count"], ascending=[False, False]).head(50)
    for _, r in top.iterrows():
        lines.append(
            f"| {r['api']} | `{r['column']}` | {float(r['null_pct']):.2f} | {int(r['null_count'])} | "
            f"{r['first_null_ts']} | {r['last_null_ts']} |"
        )
    lines.append("")
    lines.append("Hinweis: Diese Datei dokumentiert Rohquellen-Lücken. "
                 "Ob diese im finalen ML-Bundle noch sichtbar sind, hängt von der "
                 "späteren Imputation/Feature-Logik ab.")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Audit missing values in raw API parquet files.")
    p.add_argument("--raw-dir", default="data/raw", help="Raw data directory.")
    p.add_argument("--out-csv", default="data/reports/api_missingness_audit.csv", help="CSV output path.")
    p.add_argument("--out-md", default="docs/api_missingness_report.md", help="Markdown output path.")
    args = p.parse_args()

    raw_dir = (REPO_ROOT / args.raw_dir).resolve() if not Path(args.raw_dir).is_absolute() else Path(args.raw_dir)
    out_csv = (REPO_ROOT / args.out_csv).resolve() if not Path(args.out_csv).is_absolute() else Path(args.out_csv)
    out_md = (REPO_ROOT / args.out_md).resolve() if not Path(args.out_md).is_absolute() else Path(args.out_md)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dir not found: {raw_dir}")

    frames: list[pd.DataFrame] = []
    for path in sorted(raw_dir.glob("*.parquet")):
        api = API_FILE_MAP.get(path.name, f"Unknown ({path.name})")
        try:
            rep = _scan_file(path, api)
            if not rep.empty:
                frames.append(rep)
        except Exception as e:  # pragma: no cover
            print(f"[WARN] Skip {path.name}: {e}")

    if frames:
        report = pd.concat(frames, ignore_index=True, sort=False)
        report = report.sort_values(["api", "null_pct", "null_count", "column"], ascending=[True, False, False, True])
    else:
        report = pd.DataFrame(
            columns=["api", "file", "timestamp_col", "column", "rows", "null_count", "null_pct", "first_null_ts", "last_null_ts"]
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_csv, index=False)
    _write_md(report, out_md)

    print(f"[INFO] Raw dir: {raw_dir}")
    print(f"[INFO] Files scanned: {len(list(raw_dir.glob('*.parquet')))}")
    print(f"[INFO] Columns with NaNs: {len(report)}")
    print(f"[INFO] CSV written: {out_csv}")
    print(f"[INFO] Markdown written: {out_md}")


if __name__ == "__main__":
    main()
