#!/usr/bin/env python3
"""Export residual-level artifacts for reproducible forecast error analysis.

This script is intended as a production/reproducible counterpart to notebook-only
residual exploration. It writes:
- residuals.parquet (row-level)
- residual_summary_overall.csv
- residual_summary_by_lead.csv
- residual_summary_by_time_bucket.csv
- residual_summary_by_tail_regime.csv

Usage examples:
  python3 scripts/export_residual_analysis.py \
    --pred-path artifacts/model_runs/<run_id>/predictions/test_target_da_price.parquet \
    --out-dir artifacts/residual_reports/xgb_da

  python3 scripts/export_residual_analysis.py \
    --run-dir artifacts/model_runs/<run_id> --split test \
    --out-dir artifacts/residual_reports/<run_id>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _pick_first(cols: Iterable[str], candidates: list[str]) -> str | None:
    cset = set(cols)
    for c in candidates:
        if c in cset:
            return c
    return None


def _infer_prediction_files(run_dir: Path, split: str) -> list[Path]:
    files: list[Path] = []
    direct = sorted(run_dir.glob(f"{split}_*.parquet"))
    files.extend([p for p in direct if p.is_file()])
    pred_dir = run_dir / "predictions"
    if pred_dir.exists():
        files.extend(sorted(pred_dir.glob(f"*{split}*.parquet")))
    # de-duplicate while preserving order
    seen = set()
    uniq: list[Path] = []
    for p in files:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def _compute_lead_from_times(df: pd.DataFrame, ts_col: str | None) -> pd.Series:
    if "lead_time_h" in df.columns:
        return pd.to_numeric(df["lead_time_h"], errors="coerce")
    if "lead" in df.columns:
        return pd.to_numeric(df["lead"], errors="coerce")
    if "horizon_h" in df.columns:
        return pd.to_numeric(df["horizon_h"], errors="coerce")

    snap_col = _pick_first(df.columns, ["snapshot_time_utc", "snapshot_time", "forecast_time_utc"])
    tgt_col = _pick_first(df.columns, ["target_time_utc", "target_time", "timestamp_utc", "timestamp"])
    if snap_col and tgt_col:
        snap = pd.to_datetime(df[snap_col], utc=True, errors="coerce")
        tgt = pd.to_datetime(df[tgt_col], utc=True, errors="coerce")
        lead = (tgt - snap).dt.total_seconds() / 3600.0
        return pd.to_numeric(lead, errors="coerce")

    # If no lead info exists, default to 1h (explicitly marked)
    return pd.Series(np.full(len(df), 1.0), index=df.index, dtype="float64")


def _safe_mae(e: pd.Series) -> float:
    return float(pd.to_numeric(e, errors="coerce").abs().mean())


def _safe_rmse(e: pd.Series) -> float:
    v = pd.to_numeric(e, errors="coerce")
    return float(np.sqrt(np.nanmean(v.to_numpy(dtype=float) ** 2)))


def _safe_bias(e: pd.Series) -> float:
    return float(pd.to_numeric(e, errors="coerce").mean())


def _safe_std(e: pd.Series) -> float:
    return float(pd.to_numeric(e, errors="coerce").std(ddof=0))


def _residual_summary(df: pd.DataFrame) -> pd.DataFrame:
    e = df["residual"]
    out = {
        "n": int(df["residual"].notna().sum()),
        "mae": _safe_mae(e),
        "rmse": _safe_rmse(e),
        "bias": _safe_bias(e),
        "std": _safe_std(e),
        "q01": float(pd.to_numeric(e, errors="coerce").quantile(0.01)),
        "q05": float(pd.to_numeric(e, errors="coerce").quantile(0.05)),
        "q50": float(pd.to_numeric(e, errors="coerce").quantile(0.50)),
        "q95": float(pd.to_numeric(e, errors="coerce").quantile(0.95)),
        "q99": float(pd.to_numeric(e, errors="coerce").quantile(0.99)),
        "mape_percent": float(
            np.nanmean(
                np.where(
                    np.abs(df["y_true"].to_numpy(dtype=float)) > 1e-12,
                    np.abs(df["residual"].to_numpy(dtype=float) / df["y_true"].to_numpy(dtype=float)) * 100.0,
                    np.nan,
                )
            )
        ),
    }
    return pd.DataFrame([out])


def _process_one(
    pred_path: Path,
    out_dir: Path,
    *,
    y_true_col: str | None,
    y_pred_col: str | None,
    model_label: str | None,
) -> dict[str, str | int | float]:
    df = pd.read_parquet(pred_path)

    yt_col = y_true_col or _pick_first(df.columns, ["y_true", "actual", "target", "truth", "y"])
    yp_col = y_pred_col or _pick_first(df.columns, ["y_pred", "predicted_value", "prediction", "yhat", "pred"])
    ts_col = _pick_first(df.columns, ["timestamp", "target_time_utc", "timestamp_utc", "target_time"])

    if yt_col is None or yp_col is None:
        raise KeyError(
            f"Could not infer y_true/y_pred columns for {pred_path}. "
            f"Found columns head: {list(df.columns)[:20]}"
        )

    d = pd.DataFrame(index=df.index)
    d["source_file"] = str(pred_path)
    d["model_label"] = model_label or pred_path.stem
    if ts_col is not None:
        d["timestamp_utc"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    else:
        d["timestamp_utc"] = pd.NaT

    d["y_true"] = pd.to_numeric(df[yt_col], errors="coerce")
    d["y_pred"] = pd.to_numeric(df[yp_col], errors="coerce")
    d["lead_time_h"] = _compute_lead_from_times(df, ts_col)

    valid = d["y_true"].notna() & d["y_pred"].notna()
    d = d.loc[valid].copy()
    d["residual"] = d["y_true"] - d["y_pred"]
    d["abs_error"] = d["residual"].abs()
    d["sq_error"] = d["residual"] ** 2

    # Time buckets (for easy aggregation/plotting)
    if d["timestamp_utc"].notna().any():
        ts = pd.to_datetime(d["timestamp_utc"], utc=True, errors="coerce")
        d["hour_utc"] = ts.dt.hour
        d["dow_utc"] = ts.dt.dayofweek
        d["month_utc"] = ts.dt.month

    # Tail regimes by y_true distribution
    q05 = float(d["y_true"].quantile(0.05))
    q95 = float(d["y_true"].quantile(0.95))
    d["tail_regime"] = "mid_90"
    d.loc[d["y_true"] <= q05, "tail_regime"] = "bottom_5"
    d.loc[d["y_true"] >= q95, "tail_regime"] = "top_5"

    stem = pred_path.stem
    run_out = out_dir / stem
    run_out.mkdir(parents=True, exist_ok=True)

    # row-level residual artifact
    d.to_parquet(run_out / "residuals.parquet", index=False)

    # summaries
    overall = _residual_summary(d)
    overall.insert(0, "artifact", stem)
    overall.to_csv(run_out / "residual_summary_overall.csv", index=False)

    by_lead = (
        d.groupby("lead_time_h", as_index=False)
        .agg(
            n=("residual", "count"),
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda s: float(np.sqrt(np.nanmean(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float))))),
            bias=("residual", "mean"),
            std=("residual", lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0))),
        )
        .sort_values("lead_time_h")
        .reset_index(drop=True)
    )
    by_lead.to_csv(run_out / "residual_summary_by_lead.csv", index=False)

    by_tail = (
        d.groupby("tail_regime", as_index=False)
        .agg(
            n=("residual", "count"),
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda s: float(np.sqrt(np.nanmean(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float))))),
            bias=("residual", "mean"),
            std=("residual", lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0))),
        )
    )
    by_tail.to_csv(run_out / "residual_summary_by_tail_regime.csv", index=False)

    if "hour_utc" in d.columns:
        by_time = (
            d.groupby(["hour_utc", "dow_utc"], as_index=False)
            .agg(
                n=("residual", "count"),
                mae=("abs_error", "mean"),
                bias=("residual", "mean"),
                std=("residual", lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0))),
            )
        )
        by_time.to_csv(run_out / "residual_summary_by_time_bucket.csv", index=False)

    meta = {
        "artifact": stem,
        "source_file": str(pred_path),
        "n_rows": int(len(d)),
        "y_true_col": yt_col,
        "y_pred_col": yp_col,
        "timestamp_col": ts_col or "",
        "overall_mae": float(overall.loc[0, "mae"]),
        "overall_rmse": float(overall.loc[0, "rmse"]),
        "overall_bias": float(overall.loc[0, "bias"]),
    }
    (run_out / "residual_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="Export reproducible residual analysis artifacts.")
    p.add_argument("--pred-path", action="append", default=[], help="Prediction parquet path(s). Can be passed multiple times.")
    p.add_argument("--run-dir", default="", help="Model run directory for auto-discovery of prediction parquets.")
    p.add_argument("--split", default="test", help="Split tag for auto-discovery when --run-dir is used (default: test).")
    p.add_argument("--y-true-col", default="", help="Override true-value column name.")
    p.add_argument("--y-pred-col", default="", help="Override predicted-value column name.")
    p.add_argument("--model-label", default="", help="Optional model label written to residual artifacts.")
    p.add_argument("--out-dir", required=True, help="Output directory for residual artifacts.")
    args = p.parse_args()

    pred_paths: list[Path] = [Path(x) for x in args.pred_path]
    if args.run_dir:
        pred_paths.extend(_infer_prediction_files(Path(args.run_dir), split=str(args.split)))
    pred_paths = [p for p in pred_paths if p.exists() and p.is_file()]

    if not pred_paths:
        raise FileNotFoundError("No prediction parquet files found. Provide --pred-path or --run-dir.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pp in pred_paths:
        try:
            meta = _process_one(
                pp,
                out_dir,
                y_true_col=(args.y_true_col or None),
                y_pred_col=(args.y_pred_col or None),
                model_label=(args.model_label or None),
            )
            rows.append(meta)
            print(f"[OK] residual artifacts: {pp}")
        except Exception as exc:
            print(f"[WARN] skip {pp}: {exc}")

    if not rows:
        raise RuntimeError("Residual export failed for all files.")

    pd.DataFrame(rows).to_csv(out_dir / "residual_export_index.csv", index=False)
    print("[OK] index:", out_dir / "residual_export_index.csv")


if __name__ == "__main__":
    main()
