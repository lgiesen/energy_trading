#!/usr/bin/env python3
"""Compute strict gate-window benchmark metrics + stitched weekly plots.

This script enforces exact local gate snapshots and two window types:
- target_window: D+1 00:00..23:00
- bridge_window: D-1 gate_hour..23:00

It exports:
- results/gate_benchmark_metrics.csv
- results/gate_stitched_weekly_2x2_<target_variable>.png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_trading.utils.run_context import resolve_model_run_dirs


TZ_LOCAL = "Europe/Berlin"
PRED_TO_TRUTH = {
    "pred_da_price": "target_da_price",
    "pred_afrr_activation_price_pos": "target_afrr_activation_price_vwap_pos",
    "pred_afrr_activation_price_neg": "target_afrr_activation_price_vwap_neg",
    "pred_afrr_capacity_price_pos": "target_afrr_capacity_price_pos",
    "pred_afrr_capacity_price_neg": "target_afrr_capacity_price_neg",
    "pred_afrr_activation_rate_pos": "target_afrr_activation_rate_pos",
    "pred_afrr_activation_rate_neg": "target_afrr_activation_rate_neg",
}
TARGET_PRED_COLS = [
    "pred_da_price",
    "pred_afrr_activation_price_pos",
    "pred_afrr_activation_price_neg",
    "pred_afrr_capacity_price_pos",
    "pred_afrr_capacity_price_neg",
    "pred_afrr_activation_rate_pos",
    "pred_afrr_activation_rate_neg",
]


def _target_variable_from_pred_col(pred_col: str) -> str:
    if pred_col.startswith("pred_"):
        return pred_col[len("pred_") :]
    return pred_col


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def _load_summary(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "summary_metrics.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing summary_metrics.json in {run_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve_truth_path(run_dir: Path, summary: dict[str, Any]) -> Path:
    gp = str(summary.get("ground_truth_path", "")).strip()
    if gp and Path(gp).exists():
        return Path(gp)
    p = run_dir / "manifest.json"
    if p.exists():
        mp = json.loads(p.read_text(encoding="utf-8"))
        gt = str(mp.get("ground_truth", {}).get("default_path", "")).strip()
        if gt and Path(gt).exists():
            return Path(gt)
    fallback = REPO_ROOT / "data" / "features" / "all_data_features.parquet"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not resolve ground truth parquet path.")


def _available_pred_cols(summary: dict[str, Any]) -> list[str]:
    rows = summary.get("avg_metrics_by_prediction_column", [])
    cols = [str(r.get("prediction_column")) for r in rows if r.get("prediction_column")]
    return sorted(set(cols))


def _resolve_truth_col(summary: dict[str, Any], pred_col: str) -> str:
    rows = summary.get("avg_metrics_by_prediction_column", [])
    for r in rows:
        if str(r.get("prediction_column")) == pred_col and r.get("truth_column"):
            return str(r["truth_column"])
    if pred_col in PRED_TO_TRUTH:
        return PRED_TO_TRUTH[pred_col]
    raise KeyError(f"Could not resolve truth column for {pred_col}")


def _discover_long_prediction_file(run_dir: Path, split: str, pred_col: str, model_key: str) -> Path:
    pred_dir = run_dir / "predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(f"Missing predictions dir: {pred_dir}")
    cands = list(pred_dir.glob(f"*{split}*{pred_col}*long*.parquet"))
    if not cands:
        cands = list(pred_dir.glob(f"*{pred_col}*long*{split}*.parquet"))
    if not cands:
        cands = list(pred_dir.glob(f"*{pred_col}*{split}*long*.parquet"))

    mk = model_key.lower()
    if mk == "xgboost":
        cands = [p for p in cands if "xgboost" in p.name.lower()]
    elif mk == "linear":
        cands = [p for p in cands if "linear" in p.name.lower()]
    elif mk == "tft":
        cands = [p for p in cands if ("xgboost" not in p.name.lower() and "linear" not in p.name.lower())]

    if not cands:
        raise FileNotFoundError(f"No long prediction file for {pred_col} in {run_dir}")
    return sorted(cands)[0]


def _load_long_with_truth(
    run_dir: Path,
    model_key: str,
    summary: dict[str, Any],
    pred_col: str,
    split: str,
) -> pd.DataFrame:
    p = _discover_long_prediction_file(run_dir, split, pred_col, model_key=model_key)
    df = pd.read_parquet(p)
    ts_col = next((c for c in ["target_time_utc", "timestamp_utc", "timestamp"] if c in df.columns), None)
    if ts_col is None:
        raise KeyError(f"{p} has no target timestamp column.")
    pred_name = "p50" if "p50" in df.columns else ("predicted_value" if "predicted_value" in df.columns else None)
    if pred_name is None:
        raise KeyError(f"{p} has no p50/predicted_value column.")
    keep_cols = [ts_col, pred_name]
    if "snapshot_time_utc" in df.columns:
        keep_cols.append("snapshot_time_utc")
    if "lead_time_h" in df.columns:
        keep_cols.append("lead_time_h")
    for q in ("p10", "p90"):
        if q in df.columns:
            keep_cols.append(q)
    d = df[keep_cols].copy()
    d["target_time_utc"] = pd.to_datetime(d[ts_col], utc=True, errors="coerce")
    if "snapshot_time_utc" in d.columns:
        d["snapshot_time_utc"] = pd.to_datetime(d["snapshot_time_utc"], utc=True, errors="coerce")
    else:
        lead = pd.to_numeric(d["lead_time_h"], errors="coerce").fillna(0.0) if "lead_time_h" in d.columns else 0.0
        d["snapshot_time_utc"] = d["target_time_utc"] - pd.to_timedelta(lead, unit="h")
    d["y_pred"] = pd.to_numeric(d[pred_name], errors="coerce")

    truth_col = _resolve_truth_col(summary, pred_col)
    truth_path = _resolve_truth_path(run_dir, summary)
    truth = pd.read_parquet(truth_path)[["timestamp_utc", truth_col]].copy()
    truth["timestamp_utc"] = pd.to_datetime(truth["timestamp_utc"], utc=True, errors="coerce")
    truth["y_true"] = pd.to_numeric(truth[truth_col], errors="coerce")
    truth = truth.dropna(subset=["timestamp_utc"])[["timestamp_utc", "y_true"]]

    d = d.merge(truth.rename(columns={"timestamp_utc": "target_time_utc"}), on="target_time_utc", how="left")
    d = d.dropna(subset=["snapshot_time_utc", "target_time_utc", "y_pred", "y_true"]).copy()
    d["snapshot_time_utc"] = pd.to_datetime(d["snapshot_time_utc"], utc=True)
    d["target_time_utc"] = pd.to_datetime(d["target_time_utc"], utc=True)
    return d[["snapshot_time_utc", "target_time_utc", "y_pred", "y_true"]].copy()


def _strict_gate_window_filter(
    df: pd.DataFrame,
    *,
    gate_hour_local: int,
    window_type: str,
    tz: str = TZ_LOCAL,
) -> pd.DataFrame:
    if window_type not in {"target_window", "bridge_window"}:
        raise ValueError("window_type must be one of {'target_window', 'bridge_window'}")

    out = df.copy()
    out["snapshot_time_utc"] = pd.to_datetime(out["snapshot_time_utc"], utc=True, errors="coerce")
    out["target_time_utc"] = pd.to_datetime(out["target_time_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["snapshot_time_utc", "target_time_utc", "y_pred", "y_true"]).copy()
    if out.empty:
        return out

    snap_local = out["snapshot_time_utc"].dt.tz_convert(tz)
    tgt_local = out["target_time_utc"].dt.tz_convert(tz)

    m_snapshot_exact = (
        (snap_local.dt.hour == int(gate_hour_local))
        & (snap_local.dt.minute == 0)
        & (snap_local.dt.second == 0)
    )

    snap_day = snap_local.dt.floor("D")
    tgt_day = tgt_local.dt.floor("D")
    tod = (
        pd.to_timedelta(tgt_local.dt.hour, unit="h")
        + pd.to_timedelta(tgt_local.dt.minute, unit="m")
        + pd.to_timedelta(tgt_local.dt.second, unit="s")
    )

    if window_type == "target_window":
        m_day = tgt_day == (snap_day + pd.Timedelta(days=1))
        m_tod = (tod >= pd.to_timedelta("00:00:00")) & (tod <= pd.to_timedelta("23:00:00"))
    else:
        m_day = tgt_day == snap_day
        start_t = pd.to_timedelta(f"{int(gate_hour_local):02d}:00:00")
        m_tod = (tod >= start_t) & (tod <= pd.to_timedelta("23:00:00"))

    filt = out[m_snapshot_exact & m_day & m_tod].copy()
    # Keep latest snapshot per target if duplicates exist.
    filt = (
        filt.sort_values(["target_time_utc", "snapshot_time_utc"])
        .drop_duplicates(subset=["target_time_utc"], keep="last")
        .reset_index(drop=True)
    )
    return filt


def _metric_suite(df: pd.DataFrame) -> dict[str, float]:
    y = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if not bool(np.any(m)):
        return {"n": 0.0, "mae": np.nan, "rmse": np.nan}
    e = p[m] - y[m]
    return {
        "n": float(np.sum(m)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e**2))),
    }


def _pick_weeks_by_volatility(truth: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    d = truth.copy()
    d["target_time_utc"] = pd.to_datetime(d["target_time_utc"], utc=True, errors="coerce")
    d = d.dropna(subset=["target_time_utc", "y_true"]).copy()
    tloc = d["target_time_utc"].dt.tz_convert(TZ_LOCAL)
    # Monday 00:00 local week-start key.
    d["week_start_local"] = (tloc.dt.floor("D") - pd.to_timedelta(tloc.dt.weekday, unit="D")).dt.tz_localize(None)
    week_var = d.groupby("week_start_local", as_index=False)["y_true"].var().rename(columns={"y_true": "var"})
    week_var = week_var.dropna(subset=["var"]).sort_values("var")
    if week_var.empty:
        raise RuntimeError("No week variance could be computed.")
    normal = pd.Timestamp(week_var.iloc[len(week_var) // 2]["week_start_local"])
    volatile = pd.Timestamp(week_var.iloc[-1]["week_start_local"])
    return normal, volatile


def _stitched_week_prediction(
    stitched_df: pd.DataFrame,
    *,
    week_start_local_naive: pd.Timestamp,
) -> pd.DataFrame:
    # week_start_local_naive is Monday 00:00 in local time (naive)
    week_end_local_naive = week_start_local_naive + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
    d = stitched_df.copy()
    d["target_local_naive"] = d["target_time_utc"].dt.tz_convert(TZ_LOCAL).dt.tz_localize(None)
    m = (d["target_local_naive"] >= week_start_local_naive) & (d["target_local_naive"] <= week_end_local_naive)
    return d[m].copy()


def _make_stitched_plot(
    stitched_all: pd.DataFrame,
    truth_df: pd.DataFrame,
    *,
    pred_col: str,
    out_png: Path,
) -> None:
    normal_week, volatile_week = _pick_weeks_by_volatility(truth_df)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=False, sharey=False)
    sns.set_style("whitegrid")
    model_order = ["XGBoost", "TFT", "Linear"]
    colors = {"XGBoost": "#1f77b4", "TFT": "#ff7f0e", "Linear": "#2ca02c"}

    for r, gate in enumerate([8, 11]):
        for c, wk in enumerate([normal_week, volatile_week]):
            ax = axes[r, c]
            sub = stitched_all[(stitched_all["gate_hour_local"] == gate)].copy()
            sub = _stitched_week_prediction(sub, week_start_local_naive=wk)

            truth_week = _stitched_week_prediction(truth_df.copy(), week_start_local_naive=wk)
            truth_week = truth_week.sort_values("target_time_utc")
            ax.plot(
                truth_week["target_time_utc"].dt.tz_convert(TZ_LOCAL).dt.tz_localize(None),
                truth_week["y_true"],
                color="black",
                linewidth=2.5,
                label="True",
            )

            for m in model_order:
                dm = sub[sub["model"] == m].sort_values("target_time_utc")
                if dm.empty:
                    continue
                ax.plot(
                    dm["target_time_utc"].dt.tz_convert(TZ_LOCAL).dt.tz_localize(None),
                    dm["y_pred"],
                    linewidth=1.4,
                    color=colors[m],
                    label=m,
                )

            wk_lbl = "Normal" if c == 0 else "Highly Volatile"
            ax.set_title(f"{gate:02d}:00 Gate - {wk_lbl} Week")
            ax.set_xlabel("Target time (local)")
            ax.set_ylabel(pred_col)
            ax.legend(loc="best")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict gate-window benchmark metrics + stitched weekly plots.")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--results-csv", default="results/gate_benchmark_metrics.csv")
    p.add_argument("--plot-dir", default="results", help="Directory to write stitched weekly plots for all targets.")
    p.add_argument("--models", nargs="+", default=["xgboost", "tft", "linear"])
    return p


def main() -> None:
    args = _build_cli().parse_args()
    run_dirs = resolve_model_run_dirs(repo_root=REPO_ROOT, models=args.models)
    model_labels = {"xgboost": "XGBoost", "tft": "TFT", "linear": "Linear"}

    rows: list[dict[str, Any]] = []
    stitched_by_pred: dict[str, list[pd.DataFrame]] = {k: [] for k in TARGET_PRED_COLS}
    truth_by_pred: dict[str, list[pd.DataFrame]] = {k: [] for k in TARGET_PRED_COLS}

    for model_key in args.models:
        if model_key not in run_dirs:
            print(f"[WARN] run dir missing for model={model_key}; skipping")
            continue
        run_dir = run_dirs[model_key]
        summary = _load_summary(run_dir)
        pred_cols = [c for c in _available_pred_cols(summary) if c in TARGET_PRED_COLS]

        for pred_col in pred_cols:
            try:
                d = _load_long_with_truth(run_dir, model_key, summary, pred_col, split=args.split)
            except Exception as exc:
                print(f"[WARN] {model_key}/{pred_col}: {exc}")
                continue

            for gate in [8, 11]:
                for window in ["target_window", "bridge_window"]:
                    win = _strict_gate_window_filter(d, gate_hour_local=gate, window_type=window, tz=TZ_LOCAL)
                    ms = _metric_suite(win)
                    rows.append(
                        {
                            "model": model_labels.get(model_key, model_key),
                            "model_key": model_key,
                            "target_variable": _target_variable_from_pred_col(pred_col),
                            "pred_col": pred_col,
                            "gate_hour_local": int(gate),
                            "window_type": window,
                            "n": ms["n"],
                            "mae": ms["mae"],
                            "rmse": ms["rmse"],
                        }
                    )

                # Build stitched series for each target using target_window.
                st = _strict_gate_window_filter(d, gate_hour_local=gate, window_type="target_window", tz=TZ_LOCAL)
                if not st.empty:
                    st = st.copy()
                    st["model"] = model_labels.get(model_key, model_key)
                    st["gate_hour_local"] = int(gate)
                    stitched_by_pred[pred_col].append(st[["target_time_utc", "y_true", "y_pred", "model", "gate_hour_local"]])
                    truth_by_pred[pred_col].append(st[["target_time_utc", "y_true"]].copy())

    out_df = pd.DataFrame(rows).sort_values(
        ["target_variable", "pred_col", "gate_hour_local", "window_type", "model"]
    )
    out_path = REPO_ROOT / args.results_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")

    plot_dir = REPO_ROOT / args.plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)
    for pred_col in TARGET_PRED_COLS:
        stitched_parts = stitched_by_pred.get(pred_col, [])
        truth_parts = truth_by_pred.get(pred_col, [])
        if not stitched_parts or not truth_parts:
            print(f"[WARN] stitched plot skipped for {pred_col} (no rows found).")
            continue

        stitched_all = pd.concat(stitched_parts, ignore_index=True)
        truth_all = pd.concat(truth_parts, ignore_index=True).drop_duplicates(subset=["target_time_utc"]).copy()
        target_variable = _target_variable_from_pred_col(pred_col)
        out_png = plot_dir / f"gate_stitched_weekly_2x2_{target_variable}.png"
        _make_stitched_plot(
            stitched_all=stitched_all,
            truth_df=truth_all,
            pred_col=pred_col,
            out_png=out_png,
        )
        print(f"[OK] wrote {out_png}")


if __name__ == "__main__":
    main()
