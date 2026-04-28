"""Reusable analytics for benchmark tail/gate-time evaluation.

This module keeps notebook logic DRY by centralizing:
- strict gate-time + D+1 filtering
- tail-segment metrics
- quantile coverage checks
- common plots for tail calibration and quantile coverage
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MARKET_TZ = "Europe/Berlin"


@dataclass(frozen=True)
class TargetPolicy:
    category: str
    gate_hour_local: int
    tail_segments: list[tuple[str, float]]


def target_policy(pred_col: str) -> TargetPolicy:
    if pred_col == "pred_da_price":
        return TargetPolicy("DA Price", 11, [("high", 0.90), ("low", 0.10)])
    if "activation_rate" in pred_col:
        return TargetPolicy("aFRR Activation Rate", 8, [("high", 0.90)])
    if "capacity_price" in pred_col:
        return TargetPolicy("aFRR Capacity Price", 8, [("high", 0.90)])
    if "activation_price" in pred_col:
        return TargetPolicy("aFRR Activation Price", 8, [("low", 0.10)])
    # Fallback: treat as aFRR-like.
    return TargetPolicy("Other", 8, [("high", 0.90), ("low", 0.10)])


def safe_quantile(s: pd.Series, q: float) -> float:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return float("nan")
    return float(v.quantile(q))


def gate_time_dplus1_filter(
    df: pd.DataFrame,
    *,
    pred_col: str,
    snapshot_col: str = "snapshot_time_utc",
    target_col: str = "target_time_utc",
    tz: str = MARKET_TZ,
    local_tz: str | None = None,
    gate_hour_override: int | None = None,
) -> pd.DataFrame:
    """Strictly isolate decision snapshots and D+1 delivery targets.

    Protocol:
    - aFRR targets: snapshot local hour == 08:00, D+1 local date, and time <= 23:45.
    - DA target: snapshot local hour == 11:00, D+1 local date, and time <= 23:00.
    """
    out = df.copy()
    out[snapshot_col] = pd.to_datetime(out[snapshot_col], utc=True, errors="coerce")
    out[target_col] = pd.to_datetime(out[target_col], utc=True, errors="coerce")
    out = out.dropna(subset=[snapshot_col, target_col]).copy()
    if out.empty:
        return out

    tz_eff = local_tz or tz
    pol = target_policy(pred_col)
    gate_hour = int(gate_hour_override) if gate_hour_override is not None else int(pol.gate_hour_local)
    snap_local = out[snapshot_col].dt.tz_convert(tz_eff)
    tgt_local = out[target_col].dt.tz_convert(tz_eff)

    m_gate = snap_local.dt.hour == gate_hour
    m_dplus1 = tgt_local.dt.floor("D") == (snap_local.dt.floor("D") + pd.Timedelta(days=1))

    # Bound delivery day by target type.
    if pred_col == "pred_da_price":
        end_t = pd.to_timedelta("23:00:00")
    else:
        end_t = pd.to_timedelta("23:45:00")
    tod = (
        pd.to_timedelta(tgt_local.dt.hour, unit="h")
        + pd.to_timedelta(tgt_local.dt.minute, unit="m")
        + pd.to_timedelta(tgt_local.dt.second, unit="s")
    )
    m_day_window = (tod >= pd.to_timedelta("00:00:00")) & (tod <= end_t)

    filtered = out[m_gate & m_dplus1 & m_day_window].copy()
    # Keep latest snapshot in case of duplicate vintages for same target time.
    filtered = (
        filtered.sort_values([target_col, snapshot_col])
        .drop_duplicates(subset=[target_col], keep="last")
        .reset_index(drop=True)
    )
    return filtered


def build_tail_metrics(
    *,
    merged: pd.DataFrame,
    pred_col: str,
    model_ids: Iterable[str],
    model_labels: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute segment MAE/bias/spike capture per model for one target."""
    rows: list[dict[str, object]] = []
    points: list[pd.DataFrame] = []

    y_true = pd.to_numeric(merged["y_true"], errors="coerce")
    q10_true = safe_quantile(y_true, 0.10)
    q90_true = safe_quantile(y_true, 0.90)
    pol = target_policy(pred_col)

    for seg_name, _ in pol.tail_segments:
        if seg_name == "high":
            seg_thr = q90_true
            seg_mask = y_true >= seg_thr
        else:
            seg_thr = q10_true
            seg_mask = y_true <= seg_thr
        normal_mask = ~seg_mask

        for mid in model_ids:
            p50_col = f"{mid}_p50"
            if p50_col not in merged.columns:
                continue
            y = pd.to_numeric(merged["y_true"], errors="coerce")
            yhat = pd.to_numeric(merged[p50_col], errors="coerce")
            valid = y.notna() & yhat.notna()
            seg_valid = seg_mask & valid
            normal_valid = normal_mask & valid
            if int(seg_valid.sum()) == 0:
                continue

            err = yhat - y
            segment_mae = float(np.mean(np.abs(err[seg_valid])))
            normal_mae = float(np.mean(np.abs(err[normal_valid]))) if int(normal_valid.sum()) > 0 else np.nan
            tail_bias = float(np.mean(err[seg_valid]))

            spike_capture = np.nan
            pred_top20_thr = np.nan
            if seg_name == "high":
                pred_top20_thr = safe_quantile(yhat[valid], 0.80)
                if np.isfinite(pred_top20_thr):
                    spike_capture = float((yhat[seg_valid] >= pred_top20_thr).mean())

            rows.append(
                {
                    "category": pol.category,
                    "pred_col": pred_col,
                    "segment": seg_name,
                    "segment_threshold_truth": float(seg_thr) if np.isfinite(seg_thr) else np.nan,
                    "model": model_labels[mid],
                    "model_id": mid,
                    "n_segment": int(seg_valid.sum()),
                    "n_normal": int(normal_valid.sum()),
                    "segment_mae": segment_mae,
                    "normal_mae": normal_mae,
                    "tail_bias": tail_bias,
                    "spike_capture_rate": spike_capture,
                    "spike_pred_top20_threshold": float(pred_top20_thr) if np.isfinite(pred_top20_thr) else np.nan,
                }
            )

            pts = merged.loc[seg_valid, ["target_time_utc", "y_true", p50_col]].copy()
            pts = pts.rename(columns={p50_col: "y_pred"})
            pts["category"] = pol.category
            pts["pred_col"] = pred_col
            pts["segment"] = seg_name
            pts["model"] = model_labels[mid]
            pts["model_id"] = mid
            points.append(pts)

    metrics_df = pd.DataFrame(rows)
    points_df = pd.concat(points, ignore_index=True) if points else pd.DataFrame()
    return metrics_df, points_df


def aggregate_tail_metrics(tail_metrics_df: pd.DataFrame) -> pd.DataFrame:
    if tail_metrics_df.empty:
        return pd.DataFrame()
    return (
        tail_metrics_df.groupby(["category", "segment", "model"], as_index=False)
        .agg(
            n_segment=("n_segment", "sum"),
            segment_mae=("segment_mae", "mean"),
            normal_mae=("normal_mae", "mean"),
            tail_bias=("tail_bias", "mean"),
            spike_capture_rate=("spike_capture_rate", "mean"),
        )
        .sort_values(["category", "segment", "model"])
        .reset_index(drop=True)
    )


def compute_quantile_coverage(
    merged: pd.DataFrame,
    *,
    pred_col: str,
    model_ids: Iterable[str],
    quantiles: tuple[float, ...] = (0.1, 0.9),
) -> pd.DataFrame:
    """Coverage stats for p10/p90 in gate-filtered subset."""
    rows: list[dict[str, object]] = []
    y = pd.to_numeric(merged["y_true"], errors="coerce")
    pol = target_policy(pred_col)
    for mid in model_ids:
        for q in quantiles:
            qcol = f"{mid}_p{int(round(q * 100)):02d}"
            if qcol not in merged.columns:
                rows.append(
                    {
                        "pred_col": pred_col,
                        "category": pol.category,
                        "model_id": mid,
                        "quantile": q,
                        "n": 0,
                        "empirical_coverage": np.nan,
                        "coverage_gap": np.nan,
                        "note": "quantile column not available",
                    }
                )
                continue
            qv = pd.to_numeric(merged[qcol], errors="coerce")
            m = y.notna() & qv.notna()
            n = int(m.sum())
            if n == 0:
                cov = np.nan
            else:
                cov = float((y[m] <= qv[m]).mean())
            rows.append(
                {
                    "pred_col": pred_col,
                    "category": pol.category,
                    "model_id": mid,
                    "quantile": q,
                    "n": n,
                    "empirical_coverage": cov,
                    "coverage_gap": float(cov - q) if np.isfinite(cov) else np.nan,
                    "note": "",
                }
            )
    return pd.DataFrame(rows)


def plot_tail_calibration(
    points_df: pd.DataFrame,
    *,
    model_colors: dict[str, str],
    title_suffix: str,
) -> None:
    cats = ["DA Price", "aFRR Activation Rate", "aFRR Capacity Price", "aFRR Activation Price"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.reshape(-1)
    for i, cat in enumerate(cats):
        ax = axes[i]
        sub = points_df[points_df["category"] == cat].copy() if not points_df.empty else pd.DataFrame()
        if sub.empty:
            ax.text(0.5, 0.5, f"No tail points for {cat}", ha="center", va="center")
            ax.set_title(f"Tail Calibration | {cat} | {title_suffix}")
            continue
        lo = float(np.nanmin([sub["y_true"].min(), sub["y_pred"].min()]))
        hi = float(np.nanmax([sub["y_true"].max(), sub["y_pred"].max()]))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="#444444", label="Perfect")
        for mid, color in model_colors.items():
            mm = sub[sub["model_id"] == mid].copy()
            if mm.empty:
                continue
            ax.scatter(mm["y_true"], mm["y_pred"], s=12, alpha=0.30, color=color, label=mid.upper())
        ax.set_title(f"Tail Calibration | {cat} | {title_suffix}")
        ax.set_xlabel("Truth (tail subset)")
        ax.set_ylabel("Prediction (P50)")
        ax.legend(loc="best")
    plt.show()


def plot_quantile_coverage_bars(
    coverage_df: pd.DataFrame,
    *,
    title_suffix: str,
) -> None:
    if coverage_df.empty:
        return
    d = coverage_df.copy()
    d["label"] = (
        d["pred_col"].astype(str)
        + " | "
        + d["model_id"].astype(str).str.upper()
        + " | q"
        + (d["quantile"] * 100).round().astype(int).astype(str)
    )
    d = d[d["empirical_coverage"].notna()].copy()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(max(11, 0.45 * len(d)), 4.8))
    ax.bar(np.arange(len(d)), d["empirical_coverage"].to_numpy(dtype=float), alpha=0.90)
    ax.scatter(
        np.arange(len(d)),
        d["quantile"].to_numpy(dtype=float),
        marker="_",
        s=280,
        linewidths=2.0,
        label="Target coverage",
        color="#333333",
        zorder=5,
    )
    ax.set_xticks(np.arange(len(d)))
    ax.set_xticklabels(d["label"], rotation=35, ha="right")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title(f"Quantile Coverage (p10/p90) | {title_suffix}")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
