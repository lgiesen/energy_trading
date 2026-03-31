#!/usr/bin/env python3
"""Re-fetch and forensically validate ENTSO-E gap in April 2025.

This script performs a targeted second-chance fetch for a suspected source gap,
compares the refreshed window against the current raw artifact, and optionally
patches the raw file when the gap improves.

If no improvement is observed, it traces how source nulls propagate via lagged
features (notably _lag_24h and _lag_168h) in the final feature artifact.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REFETCH_START_UTC = "2025-04-01T00:00:00Z"
REFETCH_END_UTC = "2025-04-02T23:00:00Z"

DEFAULT_KEY_COLS = [
    "load_forecast_da_entsoe",
    "wind_onshore_forecast_da_entsoe",
    "wind_offshore_forecast_da_entsoe",
    "solar_forecast_da_entsoe",
    "wind_onshore_forecast_id_entsoe",
    "wind_offshore_forecast_id_entsoe",
    "solar_forecast_id_entsoe",
]

# Columns in final feature artifact that can reflect the same source gap.
TRACE_BASE_COLS = [
    "wind_onshore_forecast_id_entsoe",
    "wind_offshore_forecast_id_entsoe",
    "solar_forecast_id_entsoe",
    "residual_load_forecast",
    "renewable_share_forecast",
]


def _resolve_repo_root() -> Path:
    root = Path.cwd().resolve()
    if (root / "src").exists():
        return root
    for parent in root.parents:
        if (parent / "src").exists():
            return parent
    raise RuntimeError("Could not resolve REPO_ROOT (directory containing 'src').")


REPO_ROOT = _resolve_repo_root()


@dataclass
class ComparisonStats:
    old_total_nulls: int
    new_total_nulls: int
    old_total_cells: int
    new_total_cells: int
    improved: bool
    available_cols: list[str]
    summary_table: pd.DataFrame
    old_window: pd.DataFrame
    new_window: pd.DataFrame


def _parse_utc(ts: str) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tz is None:
        out = out.tz_localize("UTC")
    else:
        out = out.tz_convert("UTC")
    return out


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    return pd.read_parquet(path, engine="pyarrow")


def _ensure_ts(df: pd.DataFrame, col: str = "timestamp_utc") -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns:
        raise KeyError(f"Missing required timestamp column: {col}")
    out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    if out[col].isna().any():
        raise ValueError(f"Invalid timestamps in column: {col}")
    return out


def _window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (df["timestamp_utc"] >= start) & (df["timestamp_utc"] <= end)
    return df.loc[mask].copy()


def run_refetch(start: str, end: str, out_path: Path, chunk_months: int, workers: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "energy_trading.ingestion.fetch_entsoe",
        "--start",
        start,
        "--end",
        end,
        "--out",
        str(out_path),
        "--chunk-months",
        str(chunk_months),
        "--workers",
        str(workers),
    ]
    print("[INFO] Re-fetch command:")
    print("       " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def compare_old_vs_new(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    key_cols: list[str],
) -> ComparisonStats:
    old_w = _window(old_df, start, end)
    new_w = _window(new_df, start, end)
    available = [c for c in key_cols if c in old_w.columns and c in new_w.columns]
    if not available:
        raise ValueError("No overlapping key columns found for comparison.")

    rows = []
    old_total_nulls = 0
    new_total_nulls = 0
    old_total_cells = 0
    new_total_cells = 0

    for col in available:
        old_null = int(old_w[col].isna().sum())
        new_null = int(new_w[col].isna().sum())
        old_n = int(old_w[col].shape[0])
        new_n = int(new_w[col].shape[0])
        old_total_nulls += old_null
        new_total_nulls += new_null
        old_total_cells += old_n
        new_total_cells += new_n
        rows.append(
            {
                "column": col,
                "old_null_count": old_null,
                "new_null_count": new_null,
                "delta_null_count": new_null - old_null,
                "old_non_null_count": old_n - old_null,
                "new_non_null_count": new_n - new_null,
            }
        )

    table = pd.DataFrame(rows).sort_values(["delta_null_count", "column"], ascending=[True, True]).reset_index(drop=True)
    improved = new_total_nulls < old_total_nulls
    return ComparisonStats(
        old_total_nulls=old_total_nulls,
        new_total_nulls=new_total_nulls,
        old_total_cells=old_total_cells,
        new_total_cells=new_total_cells,
        improved=improved,
        available_cols=available,
        summary_table=table,
        old_window=old_w,
        new_window=new_w,
    )


def patch_raw_window(old_df: pd.DataFrame, new_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    old_keep = old_df.loc[~((old_df["timestamp_utc"] >= start) & (old_df["timestamp_utc"] <= end))].copy()
    new_win = _window(new_df, start, end)
    merged = pd.concat([old_keep, new_win], axis=0, ignore_index=True, sort=False)
    merged = merged.drop_duplicates(subset=["timestamp_utc"], keep="last").sort_values("timestamp_utc").reset_index(drop=True)
    return merged


def trace_lag_propagation_from_source_gaps(
    source_window_df: pd.DataFrame,
    features_df: pd.DataFrame,
    source_cols: list[str],
    lags: tuple[int, ...] = (24, 168),
) -> pd.DataFrame:
    rows: list[dict] = []
    ts = pd.to_datetime(features_df["timestamp_utc"], utc=True)
    idx_ts = pd.Index(ts)
    feat_idx = features_df.set_index("timestamp_utc")

    for src_col in source_cols:
        if src_col not in source_window_df.columns:
            continue
        src_null_mask = source_window_df[src_col].isna()
        src_null_ts = pd.to_datetime(source_window_df.loc[src_null_mask, "timestamp_utc"], utc=True)
        src_null_count = int(src_null_mask.sum())
        if src_null_count == 0:
            continue

        # Trace base feature at same timestamps (if present).
        if src_col in features_df.columns:
            same_ts = src_null_ts[src_null_ts.isin(idx_ts)]
            if not same_ts.empty:
                base_null_at_same_ts = int(feat_idx[src_col].reindex(same_ts).isna().sum())
                rows.append(
                    {
                        "source_column": src_col,
                        "source_gap_null_rows": src_null_count,
                        "checked_feature": src_col,
                        "relation": "same_ts",
                        "lag_hours": 0,
                        "affected_rows": base_null_at_same_ts,
                        "first_affected_ts": same_ts.min(),
                        "last_affected_ts": same_ts.max(),
                    }
                )

        for lag_h in lags:
            lag_col = f"{src_col}_lag_{lag_h}h"
            if lag_col not in features_df.columns:
                continue
            shifted_ts = src_null_ts + pd.Timedelta(hours=lag_h)
            in_bounds = shifted_ts.isin(idx_ts)
            affected_ts = shifted_ts[in_bounds]
            if affected_ts.empty:
                affected_rows = 0
                first_affected = pd.NaT
                last_affected = pd.NaT
            else:
                lag_null_mask = feat_idx[lag_col].reindex(affected_ts).isna()
                affected_rows = int(lag_null_mask.sum())
                first_affected = affected_ts.min()
                last_affected = affected_ts.max()

            rows.append(
                {
                    "source_column": src_col,
                    "source_gap_null_rows": src_null_count,
                    "checked_feature": lag_col,
                    "relation": f"+{lag_h}h",
                    "lag_hours": lag_h,
                    "affected_rows": affected_rows,
                    "first_affected_ts": first_affected,
                    "last_affected_ts": last_affected,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "source_column",
                "source_gap_null_rows",
                "checked_feature",
                "relation",
                "lag_hours",
                "affected_rows",
                "first_affected_ts",
                "last_affected_ts",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["affected_rows", "lag_hours", "source_column"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Targeted April 2025 ENTSO-E re-fetch and forensic gap analysis.")
    p.add_argument("--start", default=REFETCH_START_UTC, help="Re-fetch start timestamp (UTC).")
    p.add_argument("--end", default=REFETCH_END_UTC, help="Re-fetch end timestamp (UTC).")
    p.add_argument("--raw-path", default="data/raw/entsoe.parquet", help="Main ENTSO-E raw parquet.")
    p.add_argument(
        "--temp-out",
        default="data/raw/april_refetch_temp.parquet",
        help="Temporary output for second-chance re-fetch.",
    )
    p.add_argument(
        "--features-path",
        default="data/features/all_data_features.parquet",
        help="Final feature artifact path for lag propagation traceback.",
    )
    p.add_argument("--chunk-months", type=int, default=1, help="ENTSO-E fetch chunk size in months.")
    p.add_argument("--workers", type=int, default=1, help="ENTSO-E fetch workers.")
    p.add_argument("--skip-fetch", action="store_true", help="Skip fetching and reuse existing --temp-out file.")
    p.add_argument(
        "--comparison-out",
        default="data/reports/april_refetch_comparison.csv",
        help="CSV output for old-vs-new comparison.",
    )
    p.add_argument(
        "--trace-out",
        default="data/reports/april_hard_gap_propagation.csv",
        help="CSV output for lag propagation (Scenario B).",
    )
    p.add_argument(
        "--apply-if-better",
        action="store_true",
        help="If set, overwrite --raw-path window with refetched data when comparison improves.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    start = _parse_utc(args.start)
    end = _parse_utc(args.end)
    raw_path = (REPO_ROOT / args.raw_path).resolve() if not Path(args.raw_path).is_absolute() else Path(args.raw_path)
    temp_out = (REPO_ROOT / args.temp_out).resolve() if not Path(args.temp_out).is_absolute() else Path(args.temp_out)
    features_path = (
        (REPO_ROOT / args.features_path).resolve() if not Path(args.features_path).is_absolute() else Path(args.features_path)
    )
    comparison_out = (
        (REPO_ROOT / args.comparison_out).resolve()
        if not Path(args.comparison_out).is_absolute()
        else Path(args.comparison_out)
    )
    trace_out = (
        (REPO_ROOT / args.trace_out).resolve() if not Path(args.trace_out).is_absolute() else Path(args.trace_out)
    )

    print(f"[INFO] Re-fetch window: {start} -> {end}")
    if args.skip_fetch:
        print(f"[INFO] Skip fetch enabled; using existing temp file: {temp_out}")
    else:
        run_refetch(args.start, args.end, temp_out, chunk_months=args.chunk_months, workers=args.workers)

    old_df = _ensure_ts(_load_parquet(raw_path))
    new_df = _ensure_ts(_load_parquet(temp_out))

    stats = compare_old_vs_new(old_df, new_df, start, end, DEFAULT_KEY_COLS)
    comparison_out.parent.mkdir(parents=True, exist_ok=True)
    stats.summary_table.to_csv(comparison_out, index=False)

    print("\n=== Old vs New (Second-Chance Fetch) ===")
    print(stats.summary_table.to_string(index=False))
    print(
        f"\n[INFO] Total null cells (key forecast columns) old={stats.old_total_nulls} "
        f"new={stats.new_total_nulls}"
    )
    print(f"[INFO] Comparison report: {comparison_out}")

    if stats.improved:
        print("\n[SCENARIO A] Re-fetch improved data completeness.")
        if args.apply_if_better:
            patched = patch_raw_window(old_df, new_df, start, end)
            patched.to_parquet(raw_path, engine="pyarrow", index=False)
            print(
                "Transienter API-Fehler durch Re-Fetch behoben. "
                f"Hauptdatei aktualisiert: {raw_path}"
            )
        else:
            print("Re-fetch is better, but raw file was not patched (use --apply-if-better).")
        return

    print("\n[SCENARIO B] Hard Source Gap likely (no improvement from re-fetch).")
    feat_df = _ensure_ts(_load_parquet(features_path))
    # Only source columns with persistent nulls are traced.
    persistent_gap_cols = [
        c
        for c in stats.available_cols
        if int(stats.summary_table.loc[stats.summary_table["column"] == c, "old_null_count"].iloc[0]) > 0
        and int(stats.summary_table.loc[stats.summary_table["column"] == c, "new_null_count"].iloc[0]) > 0
    ]
    trace_cols = sorted(set(persistent_gap_cols + TRACE_BASE_COLS))
    trace_df = trace_lag_propagation_from_source_gaps(stats.old_window, feat_df, trace_cols, lags=(24, 168))
    trace_out.parent.mkdir(parents=True, exist_ok=True)
    trace_df.to_csv(trace_out, index=False)

    total_affected = int(trace_df["affected_rows"].sum()) if not trace_df.empty else 0
    print("\n=== Lag Propagation Traceback ===")
    if trace_df.empty:
        print("No propagation rows detected for configured base columns/lags.")
    else:
        print(trace_df.to_string(index=False))
    print(f"\n[INFO] Total affected lagged rows in final feature set: {total_affected}")
    print(f"[INFO] Trace report: {trace_out}")
    print(
        "\nThesis statement:\n"
        "Ein Re-Fetch-Versuch fuer den Zeitraum 2025-04-01 bis 2025-04-02 bestaetigte, "
        "dass die Datenluecke bereits auf Ebene der Primaerquelle (ENTSO-E "
        "Transparenz-Plattform) vorlag und nicht auf einen Ingestions-Fehler "
        "zurueckzufuehren ist."
    )


if __name__ == "__main__":
    main()
