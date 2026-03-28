#!/usr/bin/env python3
"""Visualize purged expanding-window CV folds.

Training: blue, Purge gap: red, Validation: green.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from energy_trading.processing.splits import PurgedTimeSeriesSplit


def _resolve_input(path: str | None) -> Path:
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Input file does not exist: {p}")
        return p
    candidates = [
        Path("data/features/all_data_features.parquet"),
        Path("data/processed/all_data_refined.parquet"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not resolve default input parquet path.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PurgedTimeSeriesSplit folds.")
    parser.add_argument("--in", dest="input_path", default=None, help="Input parquet path.")
    parser.add_argument(
        "--out",
        dest="output_path",
        default="data/reports/processed_audits/cv_folds.png",
        help="Output figure path.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-train-days", type=int, default=120)
    parser.add_argument("--test-days", type=int, default=14)
    parser.add_argument("--gap-hours", type=int, default=72)
    args = parser.parse_args()

    path = _resolve_input(args.input_path)
    df = pd.read_parquet(path)
    if "timestamp_utc" not in df.columns:
        raise KeyError("Input parquet must contain `timestamp_utc`.")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)

    splitter = PurgedTimeSeriesSplit(
        n_splits=args.n_splits,
        min_train_days=args.min_train_days,
        test_days=args.test_days,
        gap_hours=args.gap_hours,
    )
    folds = list(splitter.split_with_metadata(df, timestamps=df["timestamp_utc"]))
    if not folds:
        raise RuntimeError("No folds generated for given settings.")

    fig, ax = plt.subplots(figsize=(14, 1.2 * len(folds) + 1.5))
    for i, fold in enumerate(folds, start=1):
        y = len(folds) - i
        if len(fold.train_idx):
            x0 = df.loc[fold.train_idx[0], "timestamp_utc"]
            x1 = df.loc[fold.train_idx[-1], "timestamp_utc"]
            ax.barh(y, (x1 - x0).total_seconds() / 3600.0, left=x0, height=0.6, color="#2C7FB8", label="Train" if i == 1 else "")
        if len(fold.purge_idx):
            x0 = df.loc[fold.purge_idx[0], "timestamp_utc"]
            x1 = df.loc[fold.purge_idx[-1], "timestamp_utc"]
            ax.barh(y, (x1 - x0).total_seconds() / 3600.0, left=x0, height=0.6, color="#D7301F", label="Purge 72h" if i == 1 else "")
        if len(fold.val_idx):
            x0 = df.loc[fold.val_idx[0], "timestamp_utc"]
            x1 = df.loc[fold.val_idx[-1], "timestamp_utc"]
            ax.barh(y, (x1 - x0).total_seconds() / 3600.0, left=x0, height=0.6, color="#31A354", label="Validation" if i == 1 else "")

    ax.set_yticks(range(len(folds)))
    ax.set_yticklabels([f"Fold {k}" for k in range(len(folds), 0, -1)])
    ax.set_title("Purged Walk-Forward CV (Train / Purge / Validation)")
    ax.set_xlabel("Time (UTC)")
    ax.grid(alpha=0.2, axis="x")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
