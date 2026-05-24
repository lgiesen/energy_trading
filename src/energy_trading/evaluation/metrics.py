"""Comprehensive forecasting metrics for deterministic and quantile models."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

_QCOL_RE = re.compile(r"(?:^|_)p(?P<q>\d{1,2})$", re.IGNORECASE)
_DA_TARGET_RE = re.compile(r"target_da_price", re.IGNORECASE)
_AFRR_CAP_TARGET_RE = re.compile(r"target_afrr_capacity_price_(?:pos|neg)", re.IGNORECASE)
_STRATEGIC_INTERVAL_PAIRS: tuple[tuple[float, float], ...] = (
    (0.10, 0.90),
    (0.30, 0.70),
    (0.10, 0.30),
    (0.30, 0.50),
    (0.50, 0.70),
    (0.70, 0.90),
    (0.05, 0.95),
    (0.01, 0.99),
)


def _interval_tradeoff_score(*, picp: float | None, pinaw: float | None, target_coverage: float) -> float | None:
    """Combine calibration and sharpness into one interpretable score.

    Lower is better. Perfect calibration and zero width gives 0.
    """
    if picp is None or pinaw is None:
        return None
    if not (np.isfinite(picp) and np.isfinite(pinaw) and np.isfinite(target_coverage)):
        return None
    return float(abs(float(picp) - float(target_coverage)) + float(pinaw))


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


def _mape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-8) -> float | None:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > eps)
    if not mask.any():
        return None
    ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.mean(ape))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 2:
        return None
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = float(np.sum((yt - yp) ** 2))
    y_bar = float(np.mean(yt))
    ss_tot = float(np.sum((yt - y_bar) ** 2))
    if ss_tot <= 1e-12:
        return None
    return float(1.0 - (ss_res / ss_tot))


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


def _crps_from_quantile_losses(pinball_by_q: dict[float, float | None]) -> float | None:
    """Approximate CRPS from quantile pinball losses.

    Uses the identity CRPS = 2 * integral_0^1 pinball_tau d tau and
    trapezoidal integration over available quantiles.
    """
    pairs: list[tuple[float, float]] = []
    for q, v in sorted(pinball_by_q.items(), key=lambda kv: kv[0]):
        if v is None:
            continue
        qf = float(q)
        vf = float(v)
        if np.isfinite(qf) and np.isfinite(vf):
            pairs.append((qf, vf))
    if len(pairs) < 2:
        return None
    qs = np.asarray([p[0] for p in pairs], dtype=float)
    losses = np.asarray([p[1] for p in pairs], dtype=float)
    area = float(np.trapz(losses, qs))
    return float(2.0 * area)


def pinball_loss_by_quantile(
    y_true: pd.Series | np.ndarray,
    quantile_preds: dict[float, pd.Series | np.ndarray],
) -> dict[float, float | None]:
    """Return mean pinball loss for each quantile tau.

    Parameters
    ----------
    y_true:
        Ground truth vector.
    quantile_preds:
        Mapping {tau: predicted quantile vector}.
    """
    yt = np.asarray(pd.to_numeric(pd.Series(y_true), errors="coerce"), dtype=float)
    out: dict[float, float | None] = {}
    for tau, pred in sorted(quantile_preds.items(), key=lambda kv: kv[0]):
        yp = np.asarray(pd.to_numeric(pd.Series(pred), errors="coerce"), dtype=float)
        out[float(tau)] = _pinball_loss(yt, yp, float(tau))
    return out


def winkler_score(
    y_true: pd.Series | np.ndarray,
    y_lower: pd.Series | np.ndarray,
    y_upper: pd.Series | np.ndarray,
    *,
    alpha: float,
) -> float | None:
    """Mean Winkler score for prediction interval [L, U].

    alpha = 1 - interval_level (e.g., alpha=0.2 for an 80% interval).
    Lower score is better.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    yt = np.asarray(pd.to_numeric(pd.Series(y_true), errors="coerce"), dtype=float)
    lo = np.asarray(pd.to_numeric(pd.Series(y_lower), errors="coerce"), dtype=float)
    hi = np.asarray(pd.to_numeric(pd.Series(y_upper), errors="coerce"), dtype=float)
    m = np.isfinite(yt) & np.isfinite(lo) & np.isfinite(hi)
    if not m.any():
        return None
    yt = yt[m]
    lo = lo[m]
    hi = hi[m]
    width = hi - lo
    score = width.copy()
    below = yt < lo
    above = yt > hi
    score[below] = width[below] + (2.0 / alpha) * (lo[below] - yt[below])
    score[above] = width[above] + (2.0 / alpha) * (yt[above] - hi[above])
    return float(np.mean(score))


def prediction_interval_coverage_probability(
    y_true: pd.Series | np.ndarray,
    y_lower: pd.Series | np.ndarray,
    y_upper: pd.Series | np.ndarray,
) -> float | None:
    """PICP: fraction of observations inside [L, U]."""
    yt = np.asarray(pd.to_numeric(pd.Series(y_true), errors="coerce"), dtype=float)
    lo = np.asarray(pd.to_numeric(pd.Series(y_lower), errors="coerce"), dtype=float)
    hi = np.asarray(pd.to_numeric(pd.Series(y_upper), errors="coerce"), dtype=float)
    m = np.isfinite(yt) & np.isfinite(lo) & np.isfinite(hi)
    if not m.any():
        return None
    yt = yt[m]
    lo = lo[m]
    hi = hi[m]
    return float(np.mean((yt >= lo) & (yt <= hi)))


def prediction_interval_normalized_average_width(
    y_true: pd.Series | np.ndarray,
    y_lower: pd.Series | np.ndarray,
    y_upper: pd.Series | np.ndarray,
) -> float | None:
    """PINAW: mean interval width normalized by target range."""
    yt = np.asarray(pd.to_numeric(pd.Series(y_true), errors="coerce"), dtype=float)
    lo = np.asarray(pd.to_numeric(pd.Series(y_lower), errors="coerce"), dtype=float)
    hi = np.asarray(pd.to_numeric(pd.Series(y_upper), errors="coerce"), dtype=float)
    m = np.isfinite(yt) & np.isfinite(lo) & np.isfinite(hi)
    if not m.any():
        return None
    yt = yt[m]
    lo = lo[m]
    hi = hi[m]
    y_range = float(np.nanmax(yt) - np.nanmin(yt))
    if y_range <= 1e-12:
        return None
    return float(np.mean(hi - lo) / y_range)


def summarize_probabilistic_interval_metrics(
    df: pd.DataFrame,
    *,
    y_true_col: str,
    quantile_col_map: dict[float, str],
    lower_q: float,
    upper_q: float,
) -> dict[str, Any]:
    """Aggregate probabilistic metrics for a selected interval [q_low, q_high].

    Returns Pinball loss for all provided quantiles plus interval metrics:
    Winkler, PICP, PINAW.
    """
    if y_true_col not in df.columns:
        raise KeyError(f"Missing y_true column: {y_true_col}")
    if lower_q not in quantile_col_map or upper_q not in quantile_col_map:
        raise KeyError(f"quantile_col_map must contain {lower_q} and {upper_q}")

    y_true = _to_float_series(df[y_true_col])
    q_preds: dict[float, pd.Series] = {}
    for q, c in quantile_col_map.items():
        if c not in df.columns:
            raise KeyError(f"Missing quantile column for q={q}: {c}")
        q_preds[float(q)] = _to_float_series(df[c])

    alpha = 1.0 - float(upper_q - lower_q)
    y_lo = q_preds[float(lower_q)]
    y_hi = q_preds[float(upper_q)]

    pinball = pinball_loss_by_quantile(y_true, q_preds)
    out: dict[str, Any] = {
        "interval_lower_q": float(lower_q),
        "interval_upper_q": float(upper_q),
        "interval_level": float(upper_q - lower_q),
        "alpha": float(alpha),
        "pinball_by_quantile": {f"p{int(round(q*100)):02d}": v for q, v in pinball.items()},
        "winkler_score": winkler_score(y_true, y_lo, y_hi, alpha=alpha),
        "picp": prediction_interval_coverage_probability(y_true, y_lo, y_hi),
        "pinaw": prediction_interval_normalized_average_width(y_true, y_lo, y_hi),
    }
    return out


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
        "mape": None,
        "wmape": None,
        "r2": None,
        "directional_accuracy": None,
        "mbe": None,
        "over_prediction_ratio": None,
    }

    if yt.size > 0:
        abs_err = np.abs(yt - yp)
        sq_err = (yt - yp) ** 2
        out["mae"] = _safe_mean(abs_err)
        out["rmse"] = _safe_scalar(float(np.sqrt(np.nanmean(sq_err))))
        out["mape"] = _mape(yt, yp)
        out["wmape"] = _wmape(yt, yp)
        out["r2"] = _r2(yt, yp)
        out["directional_accuracy"] = _directional_accuracy(yt, yp)
        out["mbe"] = _mbe(yt, yp)
        out["over_prediction_ratio"] = _over_prediction_ratio(yt, yp)

    qmap = _extract_quantile_columns(df, quantile_cols)
    out["available_quantiles"] = [float(q) for q in qmap.keys()]

    pinball_by_q_float: dict[float, float | None] = {}
    for q, col in qmap.items():
        yq_s = _to_float_series(df[col])
        q_mask = yt_s.notna() & yq_s.notna()
        pinball_q = _pinball_loss(
            yt_s.loc[q_mask].to_numpy(dtype=float),
            yq_s.loc[q_mask].to_numpy(dtype=float),
            q=float(q),
        )
        pinball_by_q_float[float(q)] = pinball_q
        out[f"pinball_loss_p{int(round(q * 100)):02d}"] = pinball_q

    out["crps_quantile_approx"] = _crps_from_quantile_losses(pinball_by_q_float)

    interval_metrics_by_pair: dict[str, dict[str, float | None]] = {}
    for q_lo, q_hi in _STRATEGIC_INTERVAL_PAIRS:
        c_lo = qmap.get(float(q_lo))
        c_hi = qmap.get(float(q_hi))
        pair_key = f"p{int(round(q_lo * 100)):02d}_p{int(round(q_hi * 100)):02d}"
        pair_payload = {
            "picp": None,
            "winkler_score": None,
            "pinaw": None,
            "coverage_gap": None,
            "tradeoff_score": None,
            "interval_level": float(q_hi - q_lo),
            "alpha": float(1.0 - (q_hi - q_lo)),
        }
        if c_lo is None or c_hi is None:
            interval_metrics_by_pair[pair_key] = pair_payload
            continue
        y_lo = _to_float_series(df[c_lo])
        y_hi = _to_float_series(df[c_hi])
        m = yt_s.notna() & y_lo.notna() & y_hi.notna()
        if bool(m.any()):
            yt_m = yt_s.loc[m].to_numpy(dtype=float)
            lo = y_lo.loc[m].to_numpy(dtype=float)
            hi = y_hi.loc[m].to_numpy(dtype=float)
            alpha = float(1.0 - (q_hi - q_lo))
            pair_payload["picp"] = float(np.mean((yt_m > lo) & (yt_m < hi)))
            pair_payload["winkler_score"] = winkler_score(yt_m, lo, hi, alpha=alpha)
            pair_payload["pinaw"] = prediction_interval_normalized_average_width(yt_m, lo, hi)
            pair_payload["coverage_gap"] = float(pair_payload["picp"] - (q_hi - q_lo))
            pair_payload["tradeoff_score"] = _interval_tradeoff_score(
                picp=pair_payload["picp"],
                pinaw=pair_payload["pinaw"],
                target_coverage=float(q_hi - q_lo),
            )
        interval_metrics_by_pair[pair_key] = pair_payload
        out[f"picp_{pair_key}"] = pair_payload["picp"]
        out[f"winkler_score_{pair_key}"] = pair_payload["winkler_score"]
        out[f"pinaw_{pair_key}"] = pair_payload["pinaw"]
        out[f"coverage_gap_{pair_key}"] = pair_payload["coverage_gap"]
        out[f"tradeoff_score_{pair_key}"] = pair_payload["tradeoff_score"]

    out["interval_metrics_by_pair"] = interval_metrics_by_pair
    # Backward-compatible aliases.
    p10_p90 = interval_metrics_by_pair.get("p10_p90", {})
    out["picp_80"] = p10_p90.get("picp")
    out["winkler_score_80"] = p10_p90.get("winkler_score")
    out["pinaw_80"] = p10_p90.get("pinaw")
    out["coverage_gap_80"] = p10_p90.get("coverage_gap")
    out["tradeoff_score_80"] = p10_p90.get("tradeoff_score")
    p05_p95 = interval_metrics_by_pair.get("p05_p95", {})
    out["picp_90"] = p05_p95.get("picp")
    out["winkler_score_90"] = p05_p95.get("winkler_score")
    out["pinaw_90"] = p05_p95.get("pinaw")
    out["coverage_gap_90"] = p05_p95.get("coverage_gap")
    out["tradeoff_score_90"] = p05_p95.get("tradeoff_score")

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
