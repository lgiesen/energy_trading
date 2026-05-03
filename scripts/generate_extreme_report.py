from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_ratio(a: float, b: float) -> float | None:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= 1e-12:
        return None
    return float(a / b)


def _compute_tail_frequency_ratio(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_family, target_col, split), g in pred_df.groupby(["model_family", "target_col", "split"], dropna=False):
        y = pd.to_numeric(g["actual_value"], errors="coerce").to_numpy(dtype=float)
        p = pd.to_numeric(g["predicted_value"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(p)
        y = y[mask]
        p = p[mask]
        if y.size == 0:
            continue
        q90 = float(np.percentile(y, 90))
        q10 = float(np.percentile(y, 10))
        true_upper = float(np.mean(y >= q90))
        pred_upper = float(np.mean(p >= q90))
        true_lower = float(np.mean(y <= q10))
        pred_lower = float(np.mean(p <= q10))
        rows.append(
            {
                "model_family": str(model_family),
                "target_col": str(target_col),
                "split": str(split),
                "upper_tail_frequency_ratio": _safe_ratio(pred_upper, true_upper),
                "lower_tail_frequency_ratio": _safe_ratio(pred_lower, true_lower),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate extreme-value performance report from canonical exports.")
    p.add_argument("--metrics-path", default="artifacts/model_runs/canonical_metrics.parquet")
    p.add_argument("--predictions-path", default="artifacts/model_runs/canonical_predictions.parquet")
    p.add_argument("--out-csv", default="artifacts/benchmarks/extreme_behavior_report.csv")
    p.add_argument("--out-md", default="artifacts/benchmarks/extreme_behavior_report.md")
    args = p.parse_args()

    metrics_path = Path(args.metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"canonical metrics parquet not found: {metrics_path}")
    mdf = pd.read_parquet(metrics_path)
    needed = {"model_family", "target_col", "split", "metric_name", "metric_value"}
    missing = needed.difference(mdf.columns)
    if missing:
        raise KeyError(f"Missing columns in canonical metrics parquet: {sorted(missing)}")

    focus_metrics = {
        "spike_precision",
        "spike_recall",
        "spike_f1",
        "tail_upper_mae",
        "tail_upper_rmse",
        "tail_lower_mae",
        "tail_lower_rmse",
    }
    x = mdf[mdf["metric_name"].isin(focus_metrics)].copy()
    x["metric_value"] = pd.to_numeric(x["metric_value"], errors="coerce")
    piv = (
        x.pivot_table(
            index=["model_family", "target_col", "split"],
            columns="metric_name",
            values="metric_value",
            aggfunc="mean",
        )
        .reset_index()
    )
    piv.columns.name = None

    pred_path = Path(args.predictions_path)
    if pred_path.exists():
        pdf = pd.read_parquet(pred_path)
        pred_needed = {"model_family", "target_col", "split", "actual_value", "predicted_value"}
        if pred_needed.issubset(pdf.columns):
            freq_df = _compute_tail_frequency_ratio(pdf)
            if not freq_df.empty:
                piv = piv.merge(freq_df, on=["model_family", "target_col", "split"], how="left")

    piv = piv.sort_values(["split", "model_family", "target_col"]).reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    piv.to_csv(out_csv, index=False)

    out_md = Path(args.out_md)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Extreme Behavior Report\n\n")
        f.write(f"- rows: {len(piv)}\n")
        f.write(f"- source metrics: `{metrics_path}`\n")
        if pred_path.exists():
            f.write(f"- source predictions: `{pred_path}`\n")
        f.write("\n")
        if piv.empty:
            f.write("No extreme metrics found.\n")
        else:
            f.write(piv.to_markdown(index=False))
            f.write("\n")

    print(f"[OK] Wrote CSV: {out_csv}")
    print(f"[OK] Wrote Markdown: {out_md}")


if __name__ == "__main__":
    main()

