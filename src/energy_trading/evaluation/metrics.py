"""Comprehensive forecasting metrics for deterministic and quantile models."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

_QCOL_RE = re.compile(r"(?:^|_)p(?P<q>\d{1,2})$", re.IGNORECASE)
_DA_TARGET_RE = re.compile(r"target_da_price", re.IGNORECASE)
_AFRR_CAP_TARGET_RE = re.compile(r"target_afrr_capacity_price_(?:pos|neg)", re.IGNORECASE)


def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _safe_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    if not np.isfinite(values).any():
        return None
    return float(np.nanmean(values))


def _safe_scalar(v: float) -> float | None:
    if not np.isfinite(v):
        return None
    return float(v)


def _wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    denom = float(np.nansum(np.abs(y_true)))
    if denom <= 1e-12:
        return None
    num = float(np.nansum(np.abs(y_true - y_pred)))
    return num / denom


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if y_true.size < 2 or y_pred.size < 2:
        return None
    dy_true = np.diff(y_true)
    dy_pred = np.diff(y_pred)
    mask = np.isfinite(dy_true) & np.isfinite(dy_pred)
    if not mask.any():
        return None
    return float(np.mean(np.sign(dy_true[mask]) == np.sign(dy_pred[mask])))


def _mbe(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return None
    return float(np.mean(y_pred[mask] - y_true[mask]))


def _over_prediction_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return None
    return float(np.mean(y_pred[mask] > y_true[mask]))


def _pinball_loss(y_true: np.ndarray, y_q: np.ndarray, q: float) -> float | None:
    mask = np.isfinite(y_true) & np.isfinite(y_q)
    if not mask.any():
        return None
    e = y_true[mask] - y_q[mask]
    loss = np.maximum(q * e, (q - 1.0) * e)
    return float(np.mean(loss))


def _extract_quantile_columns(df: pd.DataFrame, explicit: dict[float, str] | None) -> dict[float, str]:
    out: dict[float, str] = {}
    if explicit:
        for q, c in explicit.items():
            if c in df.columns:
                out[float(q)] = c
    for c in df.columns:
        m = _QCOL_RE.search(c)
        if not m:
            continue
        q_int = int(m.group("q"))
        if 0 < q_int < 100:
            out.setdefault(q_int / 100.0, c)
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def compute_forecast_metrics(
    df: pd.DataFrame,
    *,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    quantile_cols: dict[float, str] | None = None,
) -> dict[str, Any]:
    """Compute deterministic + quantile metrics from a single DataFrame.

    Required columns:
    - `y_true_col`: ground-truth values
    - `y_pred_col`: point forecast (e.g., XGB point forecast or TFT p50)

    Optional quantile columns can be passed via `quantile_cols` and/or auto-
    detected from names ending in `pXX` (e.g. `y_pred_p10`, `p90`, `pred_p95`).
    """

    if y_true_col not in df.columns or y_pred_col not in df.columns:
        raise KeyError(f"Expected columns '{y_true_col}' and '{y_pred_col}' in DataFrame.")

    yt_s = _to_float_series(df[y_true_col])
    yp_s = _to_float_series(df[y_pred_col])
    base_mask = yt_s.notna() & yp_s.notna()
    yt = yt_s.loc[base_mask].to_numpy(dtype=float)
    yp = yp_s.loc[base_mask].to_numpy(dtype=float)

    out: dict[str, Any] = {
        "n_rows_input": int(len(df)),
        "n_rows_scored": int(base_mask.sum()),
        "mae": None,
        "rmse": None,
        "wmape": None,
        "directional_accuracy": None,
        "mbe": None,
        "over_prediction_ratio": None,
    }

    if yt.size > 0:
        abs_err = np.abs(yt - yp)
        sq_err = (yt - yp) ** 2
        out["mae"] = _safe_mean(abs_err)
        out["rmse"] = _safe_scalar(float(np.sqrt(np.nanmean(sq_err))))
        out["wmape"] = _wmape(yt, yp)
        out["directional_accuracy"] = _directional_accuracy(yt, yp)
        out["mbe"] = _mbe(yt, yp)
        out["over_prediction_ratio"] = _over_prediction_ratio(yt, yp)

    qmap = _extract_quantile_columns(df, quantile_cols)
    out["available_quantiles"] = [float(q) for q in qmap.keys()]

    for q, col in qmap.items():
        yq_s = _to_float_series(df[col])
        q_mask = yt_s.notna() & yq_s.notna()
        out[f"pinball_loss_p{int(round(q * 100)):02d}"] = _pinball_loss(
            yt_s.loc[q_mask].to_numpy(dtype=float),
            yq_s.loc[q_mask].to_numpy(dtype=float),
            q=float(q),
        )

    q10 = qmap.get(0.10)
    q90 = qmap.get(0.90)
    out["picp_80"] = None
    if q10 is not None and q90 is not None:
        y10 = _to_float_series(df[q10])
        y90 = _to_float_series(df[q90])
        m = yt_s.notna() & y10.notna() & y90.notna()
        if bool(m.any()):
            yt_m = yt_s.loc[m].to_numpy(dtype=float)
            lo = y10.loc[m].to_numpy(dtype=float)
            hi = y90.loc[m].to_numpy(dtype=float)
            out["picp_80"] = float(np.mean((yt_m > lo) & (yt_m < hi)))

    return out


def gate_hour_for_target(target_col: str) -> int | None:
    """Return local gate-closure hour (D-1) for supported target types."""
    if _DA_TARGET_RE.search(target_col):
        return 11
    if _AFRR_CAP_TARGET_RE.search(target_col):
        return 8
    return None


def _latest_pre_gate_rows(
    pred_long_df: pd.DataFrame,
    *,
    gate_hour_local: int,
    timezone: str = "Europe/Berlin",
    snapshot_col: str = "snapshot_time_utc",
    target_col: str = "target_time_utc",
) -> pd.DataFrame:
    if snapshot_col not in pred_long_df.columns or target_col not in pred_long_df.columns:
        raise KeyError(f"Expected '{snapshot_col}' and '{target_col}' in prediction DataFrame.")
    if pred_long_df.empty:
        return pred_long_df.head(0).copy()

    d = pred_long_df.copy()
    d[snapshot_col] = pd.to_datetime(d[snapshot_col], utc=True, errors="coerce")
    d[target_col] = pd.to_datetime(d[target_col], utc=True, errors="coerce")
    d = d[d[snapshot_col].notna() & d[target_col].notna()].copy()
    if d.empty:
        return d

    tgt_local = d[target_col].dt.tz_convert(timezone)
    gate_day_local = (tgt_local - pd.Timedelta(days=1)).dt.normalize()
    gate_cutoff_local = gate_day_local + pd.to_timedelta(gate_hour_local, unit="h")
    gate_cutoff_utc = gate_cutoff_local.dt.tz_convert("UTC")

    eligible = d[d[snapshot_col] <= gate_cutoff_utc].copy()
    if eligible.empty:
        return eligible

    # Keep the latest available snapshot at-or-before gate for each target hour.
    eligible = eligible.sort_values([target_col, snapshot_col])
    return eligible.groupby(target_col, as_index=False, sort=False).tail(1).reset_index(drop=True)


def compute_gate_closure_metrics(
    pred_long_df: pd.DataFrame,
    *,
    truth_df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str = "predicted_value",
    gate_hour_local: int,
    timezone: str = "Europe/Berlin",
    snapshot_col: str = "snapshot_time_utc",
    target_time_col: str = "target_time_utc",
) -> dict[str, Any]:
    """Compute gate-closure-sliced metrics for the latest pre-gate forecast."""
    if y_true_col not in truth_df.columns:
        raise KeyError(f"Expected truth column '{y_true_col}'.")
    if "timestamp_utc" not in truth_df.columns:
        raise KeyError("Expected 'timestamp_utc' column in truth DataFrame.")
    if y_pred_col not in pred_long_df.columns:
        raise KeyError(f"Expected prediction column '{y_pred_col}'.")

    sliced = _latest_pre_gate_rows(
        pred_long_df,
        gate_hour_local=gate_hour_local,
        timezone=timezone,
        snapshot_col=snapshot_col,
        target_col=target_time_col,
    )
    if sliced.empty:
        return {
            "gate_hour_local": int(gate_hour_local),
            "gate_timezone": timezone,
            "n_rows_gate": 0,
            "mae_gate": None,
            "rmse_gate": None,
            "acceptance_rate_gate": None,
            "metric_suite_gate": {},
        }

    truth = truth_df.loc[:, ["timestamp_utc", y_true_col]].copy()
    truth["timestamp_utc"] = pd.to_datetime(truth["timestamp_utc"], utc=True, errors="coerce")
    truth[y_true_col] = pd.to_numeric(truth[y_true_col], errors="coerce")

    merged = sliced.merge(
        truth,
        how="left",
        left_on=target_time_col,
        right_on="timestamp_utc",
    )
    merged = merged.rename(columns={y_true_col: "y_true", y_pred_col: "y_pred"})
    suite = compute_forecast_metrics(merged, y_true_col="y_true", y_pred_col="y_pred")

    # For aFRR capacity (pay-as-bid), bid acceptance occurs when bid <= clearing.
    # Here we treat y_pred as bid and y_true as realized clearing price.
    acceptance = None
    if _AFRR_CAP_TARGET_RE.search(y_true_col):
        m = pd.to_numeric(merged["y_pred"], errors="coerce").notna() & pd.to_numeric(
            merged["y_true"], errors="coerce"
        ).notna()
        if bool(m.any()):
            yp = pd.to_numeric(merged.loc[m, "y_pred"], errors="coerce").to_numpy(dtype=float)
            yt = pd.to_numeric(merged.loc[m, "y_true"], errors="coerce").to_numpy(dtype=float)
            acceptance = float(np.mean(yp <= yt))

    return {
        "gate_hour_local": int(gate_hour_local),
        "gate_timezone": timezone,
        "n_rows_gate": int(suite.get("n_rows_scored", 0) or 0),
        "mae_gate": suite.get("mae"),
        "rmse_gate": suite.get("rmse"),
        "acceptance_rate_gate": acceptance,
        "metric_suite_gate": suite,
    }
