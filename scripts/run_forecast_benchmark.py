#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_VERSION = "v1.0.0"

PRED_TARGETS = [
    "pred_da_price",
    "pred_afrr_capacity_price_pos",
    "pred_afrr_capacity_price_neg",
    "pred_afrr_activation_price_pos",
    "pred_afrr_activation_price_neg",
    "pred_afrr_activation_rate_pos",
    "pred_afrr_activation_rate_neg",
]

TARGET_DISPLAY = {
    "pred_da_price": "da_price",
    "pred_afrr_capacity_price_pos": "afrr_capacity_price_pos",
    "pred_afrr_capacity_price_neg": "afrr_capacity_price_neg",
    "pred_afrr_activation_price_pos": "afrr_activation_price_pos",
    "pred_afrr_activation_price_neg": "afrr_activation_price_neg",
    "pred_afrr_activation_rate_pos": "afrr_activation_rate_pos",
    "pred_afrr_activation_rate_neg": "afrr_activation_rate_neg",
}

TRUTH_COLUMN_CANDIDATES = {
    "pred_da_price": ["da_price", "target_da_price", "true_da_price"],
    "pred_afrr_capacity_price_pos": [
        "afrr_capacity_price_pos",
        "target_afrr_capacity_price_pos",
        "true_afrr_capacity_price_pos",
    ],
    "pred_afrr_capacity_price_neg": [
        "afrr_capacity_price_neg",
        "target_afrr_capacity_price_neg",
        "true_afrr_capacity_price_neg",
    ],
    "pred_afrr_activation_price_pos": [
        "afrr_activation_price_vwap_pos",
        "afrr_activation_price_pos",
        "target_afrr_activation_price_pos",
        "target_afrr_activation_price_vwap_pos",
        "true_afrr_activation_price_pos",
    ],
    "pred_afrr_activation_price_neg": [
        "afrr_activation_price_vwap_neg",
        "afrr_activation_price_neg",
        "target_afrr_activation_price_neg",
        "target_afrr_activation_price_vwap_neg",
        "true_afrr_activation_price_neg",
    ],
    "pred_afrr_activation_rate_pos": [
        "activation_rate_phys_pos",
        "afrr_activation_rate_pos",
        "target_afrr_activation_rate_pos",
        "true_afrr_activation_rate_pos",
    ],
    "pred_afrr_activation_rate_neg": [
        "activation_rate_phys_neg",
        "afrr_activation_rate_neg",
        "target_afrr_activation_rate_neg",
        "true_afrr_activation_rate_neg",
    ],
}

QUANTILES_REQUIRED = ["p10", "p30", "p50", "p70", "p90"]
ALL_QUANTILE_COLUMNS = [f"p{i:02d}" for i in range(1, 100)]


@dataclass
class JoinedFrame:
    model: str
    pred_target: str
    split: str
    prediction_file: Path
    truth_file: Path
    resolved_truth_col: str
    joined: pd.DataFrame
    inventory_row: dict[str, object]


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def _resolve_latest_manifest_pointer(path: Path) -> Path:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "manifest_path" in data or "manifest_path_abs" in data:
        rel = data.get("manifest_path") or data.get("manifest_path_abs")
        cand = (path.parent / str(rel)).resolve()
        if cand.exists():
            return cand
    run_id = data.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        cand = (path.parent / run_id / "manifest.json").resolve()
        if cand.exists():
            return cand
    if "bundles" in data:
        return path
    raise KeyError(f"Could not resolve run manifest from {path}. keys={list(data.keys())}")


def _resolve_model_name(run_manifest: dict, path: Path) -> str:
    mt = str(run_manifest.get("training", {}).get("model_type", "")).strip().lower()
    if mt:
        if "xg" in mt:
            return "xgb"
        return mt
    nm = path.name.lower()
    if "xgb" in nm or "xgboost" in nm:
        return "xgb"
    if "linear" in nm:
        return "linear_torch"
    if "tft" in nm:
        return "tft"
    return path.stem


def _resolve_truth_col(truth_df: pd.DataFrame, pred_target: str) -> str:
    cands = TRUTH_COLUMN_CANDIDATES[pred_target]
    matches = [c for c in cands if c in truth_df.columns]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        nearby = get_close_matches(pred_target, list(truth_df.columns), n=12, cutoff=0.3)
        raise KeyError(
            f"No truth-column match for {pred_target}. Tried={cands}. Nearby={nearby}"
        )
    raise ValueError(
        f"Ambiguous truth-column mapping for {pred_target}. matches={matches}. candidates={cands}"
    )


def _resolve_bundle_for_pred_target(pred_target: str) -> str:
    return "da" if pred_target == "pred_da_price" else "afrr"


def _resolve_prediction_long_paths(run_manifest_path: Path, run_manifest: dict, split: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    bundles = run_manifest.get("bundles", {})
    for bundle_name, bundle in bundles.items():
        if not isinstance(bundle, dict):
            continue
        pl = bundle.get("predictions_long", {})
        if not isinstance(pl, dict) or split not in pl or not isinstance(pl[split], dict):
            continue
        for pred_key, rel in pl[split].items():
            if pred_key not in PRED_TARGETS:
                continue
            p = (run_manifest_path.parent / str(rel)).resolve()
            out[pred_key] = p
    return out


def _resolve_truth_table_path(run_manifest_path: Path, run_manifest: dict, split: str, bundle_name: str, truth_override: Path | None) -> Path:
    if truth_override is not None:
        return truth_override
    bundle_truth = (Path.cwd() / "data" / "model_input" / bundle_name / f"{split}.parquet").resolve()
    if bundle_truth.exists():
        return bundle_truth
    gt = run_manifest.get("ground_truth", {})
    p = gt.get("default_path") if isinstance(gt, dict) else None
    if not isinstance(p, str) or not p.strip():
        raise KeyError(f"Missing ground_truth.default_path in manifest: {run_manifest_path}")
    return (run_manifest_path.parent / p).resolve()


def _quantile_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c in ALL_QUANTILE_COLUMNS])


def _pinball(y: np.ndarray, yhat: np.ndarray, q: float) -> float:
    e = y - yhat
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _winkler(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    width = hi - lo
    score = width.copy()
    below = y < lo
    above = y > hi
    score[below] += (2.0 / alpha) * (lo[below] - y[below])
    score[above] += (2.0 / alpha) * (y[above] - hi[above])
    return float(np.mean(score))


def _approx_crps_quantiles(y: np.ndarray, quant_preds: dict[str, np.ndarray]) -> float:
    vals = []
    for qcol, arr in quant_preds.items():
        q = int(qcol[1:]) / 100.0
        vals.append(_pinball(y, arr, q))
    return float(2.0 * np.mean(vals)) if vals else float("nan")


def _directional_accuracy(y: np.ndarray, yhat: np.ndarray) -> float:
    if len(y) < 2:
        return float("nan")
    dy = np.sign(np.diff(y))
    dh = np.sign(np.diff(yhat))
    return float(np.mean(dy == dh))


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _join_one(
    *,
    model: str,
    pred_target: str,
    split: str,
    pred_path: Path,
    truth_path: Path,
    coverage_threshold: float,
) -> JoinedFrame:
    pred = _read_table(pred_path)
    req = {"snapshot_time_utc", "target_time_utc", "lead_time_h", "predicted_value"}
    missing = sorted(list(req - set(pred.columns)))
    if missing:
        raise KeyError(f"{model}/{pred_target}/{split}: missing required prediction columns {missing}")
    qcols = _quantile_cols(pred)
    miss_q = [q for q in QUANTILES_REQUIRED if q not in qcols]

    truth = _read_table(truth_path)
    ts_truth = "timestamp_utc" if "timestamp_utc" in truth.columns else ("target_time_utc" if "target_time_utc" in truth.columns else None)
    if ts_truth is None:
        raise KeyError(f"{model}/{pred_target}/{split}: truth missing timestamp_utc/target_time_utc: {truth_path}")

    truth_col = _resolve_truth_col(truth, pred_target)

    p = pred.copy()
    p["snapshot_time_utc"] = pd.to_datetime(p["snapshot_time_utc"], utc=True, errors="coerce")
    p["target_time_utc"] = pd.to_datetime(p["target_time_utc"], utc=True, errors="coerce")
    p["lead_time_h"] = _safe_numeric(p["lead_time_h"])
    p["predicted_value"] = _safe_numeric(p["predicted_value"])
    for q in qcols:
        p[q] = _safe_numeric(p[q])
    bad_ts = p["snapshot_time_utc"].isna() | p["target_time_utc"].isna()
    if bad_ts.any():
        raise ValueError(f"{model}/{pred_target}/{split}: invalid prediction timestamps in {pred_path.name}: {int(bad_ts.sum())}")

    t = truth.copy()
    t["timestamp_utc"] = pd.to_datetime(t[ts_truth], utc=True, errors="coerce")
    t = t.dropna(subset=["timestamp_utc"]).copy()
    t[truth_col] = _safe_numeric(t[truth_col])
    t = t[["timestamp_utc", truth_col]].drop_duplicates("timestamp_utc", keep="last")

    truth_min_ts = t["timestamp_utc"].min()
    truth_max_ts = t["timestamp_utc"].max()
    p_all = p
    p = p[(p["target_time_utc"] >= truth_min_ts) & (p["target_time_utc"] <= truth_max_ts)].copy()
    dropped_outside_truth_window = int(len(p_all) - len(p))
    if p.empty:
        raise ValueError(
            f"{model}/{pred_target}/{split}: no prediction rows overlap truth window "
            f"[{truth_min_ts}, {truth_max_ts}]"
        )

    joined = p.merge(t, left_on="target_time_utc", right_on="timestamp_utc", how="left", validate="m:1")
    joined = joined.rename(columns={truth_col: "y_true"})

    pred_rows = int(len(p))
    joined_rows = int(len(joined))
    missing_truth = int(joined["y_true"].isna().sum())
    coverage = (pred_rows - missing_truth) / pred_rows if pred_rows else 0.0

    status = "ok"
    if miss_q:
        status = "missing_required_quantiles"
    if coverage < coverage_threshold:
        status = "coverage_fail"

    inv = {
        "model": model,
        "target": pred_target,
        "split": split,
        "prediction_file": str(pred_path),
        "truth_file": str(truth_path),
        "pred_rows_raw": int(len(p_all)),
        "pred_rows": pred_rows,
        "dropped_pred_rows_outside_truth_window": dropped_outside_truth_window,
        "truth_rows": int(len(t)),
        "joined_rows": joined_rows,
        "join_coverage_pct": coverage * 100.0,
        "missing_truth_rows": missing_truth,
        "resolved_truth_col": truth_col,
        "available_quantiles": ",".join(qcols),
        "missing_required_quantiles": ",".join(miss_q),
        "min_snapshot_time_utc": p["snapshot_time_utc"].min(),
        "max_snapshot_time_utc": p["snapshot_time_utc"].max(),
        "min_target_time_utc": p["target_time_utc"].min(),
        "max_target_time_utc": p["target_time_utc"].max(),
        "min_truth_timestamp_utc": t["timestamp_utc"].min(),
        "max_truth_timestamp_utc": t["timestamp_utc"].max(),
        "lead_time_min": float(p["lead_time_h"].min()) if len(p) else float("nan"),
        "lead_time_max": float(p["lead_time_h"].max()) if len(p) else float("nan"),
        "duplicate_key_count": int(p.duplicated(subset=["snapshot_time_utc", "target_time_utc", "lead_time_h"]).sum()),
        "status": status,
    }

    if coverage < coverage_threshold:
        miss_examples = joined.loc[joined["y_true"].isna(), "target_time_utc"].dropna().astype(str).head(10).tolist()
        raise ValueError(
            f"{model}/{pred_target}/{split}: coverage {coverage:.4%} < {coverage_threshold:.4%} "
            f"(missing {missing_truth}/{pred_rows}), examples={miss_examples}"
        )
    if miss_q:
        raise KeyError(f"{model}/{pred_target}/{split}: missing required quantiles {miss_q}")

    joined["model"] = model
    joined["target"] = pred_target
    joined["split"] = split
    joined["prediction_file"] = str(pred_path)
    joined["truth_file"] = str(truth_path)
    return JoinedFrame(model=model, pred_target=pred_target, split=split, prediction_file=pred_path, truth_file=truth_path, resolved_truth_col=truth_col, joined=joined, inventory_row=inv)


def _gate_filter(df: pd.DataFrame, pred_target: str) -> tuple[pd.DataFrame, str]:
    # heuristic, explicit status output
    if pred_target == "pred_da_price":
        g = df[df["snapshot_time_utc"].dt.hour == 11].copy()
        return g, "ok"
    if "capacity_price" in pred_target:
        g = df[df["snapshot_time_utc"].dt.hour == 8].copy()
        return g, "ok"
    if "activation" in pred_target:
        g = df[df["lead_time_h"].between(1, 6, inclusive="both")].copy()
        return g, "ok"
    return df.copy(), "ambiguous"


def _core_metrics(df: pd.DataFrame) -> dict[str, float]:
    y = _safe_numeric(df["y_true"]).to_numpy(dtype=float)
    p50 = _safe_numeric(df["p50"]).to_numpy(dtype=float)
    out = {
        "n": float(len(df)),
        "mae_p50": float(np.mean(np.abs(y - p50))),
        "rmse_p50": _rmse(y, p50),
        "bias_p50": float(np.mean(p50 - y)),
        "wmape_p50": float(np.sum(np.abs(y - p50)) / max(np.sum(np.abs(y)), 1e-12)),
        "directional_accuracy_p50": _directional_accuracy(y, p50),
    }
    qpreds = {q: _safe_numeric(df[q]).to_numpy(dtype=float) for q in _quantile_cols(df)}
    for q, arr in qpreds.items():
        out[f"pinball_{q}"] = _pinball(y, arr, int(q[1:]) / 100.0)
    out["mean_pinball_loss"] = float(np.mean([v for k, v in out.items() if k.startswith("pinball_")]))
    out["approx_crps"] = _approx_crps_quantiles(y, qpreds)

    p10 = _safe_numeric(df["p10"]).to_numpy(dtype=float)
    p30 = _safe_numeric(df["p30"]).to_numpy(dtype=float)
    p70 = _safe_numeric(df["p70"]).to_numpy(dtype=float)
    p90 = _safe_numeric(df["p90"]).to_numpy(dtype=float)
    out["coverage_p10_p90"] = float(np.mean((y >= p10) & (y <= p90)))
    out["coverage_p30_p70"] = float(np.mean((y >= p30) & (y <= p70)))
    out["winkler_p10_p90"] = _winkler(y, p10, p90, alpha=0.2)
    out["winkler_p30_p70"] = _winkler(y, p30, p70, alpha=0.4)

    quant_cols = _quantile_cols(df)
    cross_viol = np.zeros(len(df), dtype=float)
    if quant_cols:
        vals = np.vstack([_safe_numeric(df[q]).to_numpy(dtype=float) for q in quant_cols]).T
        for i in range(len(quant_cols) - 1):
            cross_viol += np.maximum(vals[:, i] - vals[:, i + 1], 0.0)
    out["quantile_crossing_rate"] = float(np.mean(cross_viol > 0.0))
    out["max_crossing_violation"] = float(np.max(cross_viol)) if len(cross_viol) else 0.0
    return out


def _tail_metrics(df: pd.DataFrame, pred_target: str) -> list[dict[str, object]]:
    y = _safe_numeric(df["y_true"])
    p50 = _safe_numeric(df["p50"])
    base_mae = float(np.mean(np.abs(y - p50)))

    rows: list[dict[str, object]] = []
    for tail, qs in [("upper", [0.8, 0.9, 0.95]), ("lower", [0.2, 0.1, 0.05])]:
        for q in qs:
            thr = float(y.quantile(q))
            if tail == "upper":
                m = y >= thr
                event_pred = p50 >= thr
            else:
                m = y <= thr
                event_pred = p50 <= thr
            if m.sum() == 0:
                continue
            y_t = y[m]
            p_t = p50[m]
            tp = float((m & event_pred).sum())
            fp = float((~m & event_pred).sum())
            fn = float((m & ~event_pred).sum())
            precision = tp / max(tp + fp, 1e-12)
            recall = tp / max(tp + fn, 1e-12)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            row = {
                "model": df["model"].iloc[0],
                "target": pred_target,
                "tail_side": tail,
                "tail_quantile": q,
                "n_tail": int(m.sum()),
                "tail_mae": float(np.mean(np.abs(y_t - p_t))),
                "tail_rmse": _rmse(y_t.to_numpy(float), p_t.to_numpy(float)),
                "tail_bias": float(np.mean(p_t - y_t)),
                "tail_mae_overall_mae_ratio": float(np.mean(np.abs(y_t - p_t)) / max(base_mae, 1e-12)),
                "high_event_recall": recall,
                "high_event_precision": precision,
                "high_event_f1": f1,
                "spike_capture_rate": recall,
                "avg_predicted_quantile_on_extremes": float(np.mean(p_t)),
                "value_weighted_abs_error": float(np.mean(np.abs(y_t - p_t) * np.abs(y_t))),
            }
            rows.append(row)
    return rows


def _joint_event_metrics(df_by_target: dict[str, pd.DataFrame], model: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def _join_on_ts(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        return a[["target_time_utc", "y_true", "p50"]].rename(columns={"y_true": "y_a", "p50": "p_a"}).merge(
            b[["target_time_utc", "y_true", "p50"]].rename(columns={"y_true": "y_b", "p50": "p_b"}),
            on="target_time_utc",
            how="inner",
        )

    combos = [
        ("pred_afrr_capacity_price_pos", "pred_afrr_activation_price_pos", "pos_cap_and_pos_act"),
        ("pred_afrr_activation_price_pos", "pred_afrr_activation_rate_pos", "pos_act_price_and_rate"),
        ("pred_afrr_capacity_price_pos", "pred_afrr_activation_rate_pos", "pos_cap_and_rate"),
        ("pred_afrr_capacity_price_neg", "pred_afrr_activation_price_neg", "neg_cap_and_neg_act"),
        ("pred_afrr_activation_rate_neg", "pred_afrr_activation_price_neg", "neg_rate_and_neg_act"),
    ]
    for ta, tb, name in combos:
        if ta not in df_by_target or tb not in df_by_target:
            continue
        j = _join_on_ts(df_by_target[ta], df_by_target[tb])
        if j.empty:
            continue
        thr_a = float(j["y_a"].quantile(0.9))
        # for neg activation price, favorable = low (more negative)
        if "activation_price_neg" in tb:
            event_true_b = j["y_b"] <= float(j["y_b"].quantile(0.1))
            event_pred_b = j["p_b"] <= float(j["y_b"].quantile(0.1))
        else:
            event_true_b = j["y_b"] >= float(j["y_b"].quantile(0.9))
            event_pred_b = j["p_b"] >= float(j["y_b"].quantile(0.9))
        event_true_a = j["y_a"] >= thr_a
        event_pred_a = j["p_a"] >= thr_a
        et = event_true_a & event_true_b
        ep = event_pred_a & event_pred_b
        tp = float((et & ep).sum())
        fp = float((~et & ep).sum())
        fn = float((et & ~ep).sum())
        prec = tp / max(tp + fp, 1e-12)
        rec = tp / max(tp + fn, 1e-12)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        value = np.abs(j["y_a"] * j["y_b"])
        rows.append(
            {
                "model": model,
                "joint_event": name,
                "joint_event_recall": rec,
                "joint_event_precision": prec,
                "joint_event_f1": f1,
                "captured_event_value_mean": float(value[et & ep].mean()) if (et & ep).any() else 0.0,
                "missed_event_value_mean": float(value[et & ~ep].mean()) if (et & ~ep).any() else 0.0,
                "missed_high_value_event_count": int((et & ~ep).sum()),
            }
        )
    return rows


def _quantile_pair_map(da_role: str) -> pd.DataFrame:
    rows = []
    for scenario, lo, hi in [("p50_p50", "p50", "p50"), ("p70_p90", "p70", "p90")]:
        for t in PRED_TARGETS:
            if t == "pred_da_price":
                q = {"low": lo, "high": hi, "mid": "p50"}[da_role]
            elif t.endswith("_neg"):
                q = lo
            else:
                q = hi
            rows.append({
                "scenario": scenario,
                "target": t,
                "selected_quantile": q,
                "risk_note": "central" if scenario == "p50_p50" else "aggressive_tail_seeking",
            })
    return pd.DataFrame(rows)


def _quantile_pair_diag(df: pd.DataFrame, qmap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, target), g in df.groupby(["model", "target"], dropna=False):
        for _, m in qmap[qmap["target"] == target].iterrows():
            q = str(m["selected_quantile"])
            y = _safe_numeric(g["y_true"]).to_numpy(float)
            yhat = _safe_numeric(g[q]).to_numpy(float)
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "scenario": m["scenario"],
                    "selected_quantile": q,
                    "selected_quantile_pinball": _pinball(y, yhat, int(q[1:]) / 100.0),
                    "selected_quantile_bias": float(np.mean(yhat - y)),
                    "selected_quantile_hit_rate": float(np.mean(y <= yhat)),
                    "overprediction_frequency": float(np.mean(yhat > y)),
                    "underprediction_frequency": float(np.mean(yhat < y)),
                    "tail_event_capture": float(np.mean(yhat[y >= np.quantile(y, 0.9)] >= np.quantile(y, 0.9))) if np.sum(y >= np.quantile(y, 0.9)) else 0.0,
                    "risk_note": m["risk_note"],
                }
            )
    return pd.DataFrame(rows)


def _score_models(core: pd.DataFrame, gate: pd.DataFrame, tail: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    weights = {
        "p50_accuracy": 0.25,
        "gate_time": 0.20,
        "tail": 0.25,
        "calibration": 0.15,
        "crossing": 0.05,
        "value_event": 0.10,
    }
    out = []
    for (model, target), g in core.groupby(["model", "target"], dropna=False):
        row = g.iloc[0]
        gate_row = gate[(gate["model"] == model) & (gate["target"] == target)]
        tail_row = tail[(tail["model"] == model) & (tail["target"] == target)]
        p50_score = 1.0 / max(float(row["mae_p50"]), 1e-9)
        gate_score = 1.0 / max(float(gate_row["mae_p50"].iloc[0]) if not gate_row.empty else float(row["mae_p50"]), 1e-9)
        tail_score = 1.0 / max(float(tail_row["tail_mae"].mean()) if not tail_row.empty else float(row["mae_p50"]), 1e-9)
        calib_score = max(0.0, 1.0 - abs(float(row["coverage_p10_p90"]) - 0.8))
        crossing_penalty = float(row.get("quantile_crossing_rate", 0.0))
        value_event = 1.0 / max(float(tail_row["value_weighted_abs_error"].mean()) if not tail_row.empty else float(row["mae_p50"]), 1e-9)
        final = (
            weights["p50_accuracy"] * p50_score
            + weights["gate_time"] * gate_score
            + weights["tail"] * tail_score
            + weights["calibration"] * calib_score
            + weights["value_event"] * value_event
            - weights["crossing"] * crossing_penalty
        )
        out.append(
            {
                "model": model,
                "target": target,
                "p50_error_score": p50_score,
                "gate_time_score": gate_score,
                "tail_score": tail_score,
                "calibration_score": calib_score,
                "crossing_penalty": crossing_penalty,
                "value_event_score": value_event,
                "final_composite_score": final,
            }
        )
    return pd.DataFrame(out), weights


def _recommend(scores: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows = []
    for target, g in scores.groupby("target", dropna=False):
        g2 = g.sort_values("final_composite_score", ascending=False)
        b = g2.iloc[0]
        rows.append(
            {
                "target": target,
                "recommended_model": b["model"],
                "runner/model_id": b["model"],
                "reason": "highest composite forecast benchmark score",
                "p50_score": b["p50_error_score"],
                "gate_score": b["gate_time_score"],
                "tail_score": b["tail_score"],
                "calibration_score": b["calibration_score"],
                "main_weakness": "see per-target tail/calibration metrics",
                "thesis_risk_note": "forecast-side evidence only; not direct PnL guarantee",
                "acceptable_for_simulation": 1,
            }
        )
    rec = pd.DataFrame(rows)
    try:
        md = rec.to_markdown(index=False)
    except ImportError:
        # Keep benchmark runnable in minimal envs without optional tabulate dependency.
        md = rec.to_csv(index=False)
    return rec, md


def _plot_target(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    d = df.sort_values("target_time_utc").copy()
    y = _safe_numeric(d["y_true"]).to_numpy(float)
    p50 = _safe_numeric(d["p50"]).to_numpy(float)
    p10 = _safe_numeric(d["p10"]).to_numpy(float)
    p90 = _safe_numeric(d["p90"]).to_numpy(float)
    lead = _safe_numeric(d["lead_time_h"]).to_numpy(float)

    # calibration-ish coverage bar
    cov_10_90 = np.mean((y >= p10) & (y <= p90))
    cov_30_70 = np.mean((_safe_numeric(d["p30"]).to_numpy(float) <= y) & (y <= _safe_numeric(d["p70"]).to_numpy(float)))
    plt.figure(figsize=(6, 4))
    plt.bar(["p10-p90", "p30-p70"], [cov_10_90, cov_30_70])
    plt.ylim(0, 1)
    plt.title("Empirical Coverage")
    plt.tight_layout()
    plt.savefig(out_dir / "calibration.png", dpi=140)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(y, p50, s=8, alpha=0.4)
    lo = float(np.nanmin([np.nanmin(y), np.nanmin(p50)]))
    hi = float(np.nanmax([np.nanmax(y), np.nanmax(p50)]))
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.title("Tail Scatter y_true vs p50")
    plt.tight_layout()
    plt.savefig(out_dir / "tail_scatter.png", dpi=140)
    plt.close()

    k = min(200, len(d))
    plt.figure(figsize=(10, 4))
    plt.plot(d["target_time_utc"].iloc[:k], y[:k], label="truth", lw=1)
    plt.plot(d["target_time_utc"].iloc[:k], p50[:k], label="p50", lw=1)
    plt.fill_between(d["target_time_utc"].iloc[:k], p10[:k], p90[:k], alpha=0.2, label="p10-p90")
    plt.legend()
    plt.title("Prediction vs Truth (sample)")
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_vs_truth_sample.png", dpi=140)
    plt.close()

    plt.figure(figsize=(6, 4))
    resid = p50 - y
    plt.scatter(lead, resid, s=8, alpha=0.3)
    plt.axhline(0, color="k", lw=1)
    plt.title("Residual by lead")
    plt.tight_layout()
    plt.savefig(out_dir / "residual_by_lead.png", dpi=140)
    plt.close()


def _parse_quantile_pairs(text: str) -> list[tuple[str, str]]:
    out = []
    for p in text.split(","):
        p = p.strip()
        if not p:
            continue
        lo, hi = p.split("-", 1)
        out.append((lo.strip().lower(), hi.strip().lower()))
    return out


def _run(args: argparse.Namespace) -> int:
    if args.models:
        model_filter = {m.strip() for m in args.models.split(",") if m.strip()}
    else:
        model_filter = set()
    target_filter = {t.strip() for t in args.targets.split(",") if t.strip()} if args.targets else set()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else (Path("artifacts/forecast_benchmark") / ts)

    if out_dir.exists() and args.clean_output:
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    all_joined: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, object]] = []
    mapping_audit_rows: list[dict[str, object]] = []
    warnings: list[str] = []

    for mpath_str in args.manifests:
        latest = Path(mpath_str).resolve()
        run_manifest_path = _resolve_latest_manifest_pointer(latest)
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        model = _resolve_model_name(run_manifest, run_manifest_path)
        if model_filter and model not in model_filter:
            continue

        pred_paths = _resolve_prediction_long_paths(run_manifest_path, run_manifest, args.split)
        for pred_target, pred_path in pred_paths.items():
            if target_filter and pred_target not in target_filter:
                continue
            bundle = _resolve_bundle_for_pred_target(pred_target)
            truth_path = _resolve_truth_table_path(run_manifest_path, run_manifest, args.split, bundle, Path(args.truth).resolve() if args.truth else None)

            jf = _join_one(
                model=model,
                pred_target=pred_target,
                split=args.split,
                pred_path=pred_path,
                truth_path=truth_path,
                coverage_threshold=float(args.coverage_threshold),
            )
            all_joined.append(jf.joined)
            inventory_rows.append(jf.inventory_row)
            mapping_audit_rows.append(
                {
                    "model": model,
                    "target": pred_target,
                    "split": args.split,
                    "prediction_file": str(pred_path),
                    "truth_file": str(truth_path),
                    "resolved_truth_col": jf.resolved_truth_col,
                    "join_coverage_pct": jf.inventory_row["join_coverage_pct"],
                    "status": jf.inventory_row["status"],
                }
            )

    if not all_joined:
        raise RuntimeError("No joined benchmark rows produced.")

    joined_df = pd.concat(all_joined, ignore_index=True)
    inv_df = pd.DataFrame(inventory_rows)
    map_df = pd.DataFrame(mapping_audit_rows)
    inv_df.to_csv(out_dir / "data_inventory.csv", index=False)
    map_df.to_csv(out_dir / "truth_mapping_audit.csv", index=False)

    core_rows = []
    gate_rows = []
    tail_rows = []
    for (model, target, split), g in joined_df.groupby(["model", "target", "split"], dropna=False):
        m = _core_metrics(g)
        core_rows.append({"model": model, "target": target, "split": split, **m})

        gg, status = _gate_filter(g, target)
        if gg.empty:
            gate_rows.append({"model": model, "target": target, "split": split, "status": "ambiguous_or_empty", "n": 0})
        else:
            gm = _core_metrics(gg)
            gate_rows.append({"model": model, "target": target, "split": split, "status": status, **gm})

        tail_rows.extend(_tail_metrics(g, target))

        if args.plots:
            _plot_target(g, out_dir / "figures" / model / target)
            (out_dir / "targets" / target / "figures").mkdir(parents=True, exist_ok=True)
            _plot_target(g, out_dir / "targets" / target / "figures")

    core_df = pd.DataFrame(core_rows)
    gate_df = pd.DataFrame(gate_rows)
    tail_df = pd.DataFrame(tail_rows)

    core_df.to_csv(out_dir / "forecast_metrics_probabilistic.csv", index=False)
    gate_df.to_csv(out_dir / "gate_time_forecast_metrics.csv", index=False)
    tail_df.to_csv(out_dir / "tail_performance_value_events.csv", index=False)

    joint_rows = []
    for model, g_model in joined_df.groupby("model", dropna=False):
        by_target = {t: gt.copy() for t, gt in g_model.groupby("target", dropna=False)}
        joint_rows.extend(_joint_event_metrics(by_target, model))
    joint_df = pd.DataFrame(joint_rows)
    joint_df.to_csv(out_dir / "joint_value_event_diagnostics.csv", index=False)

    qmap = _quantile_pair_map(args.da_quantile_role)
    qmap.to_csv(out_dir / "quantile_pair_mapping.csv", index=False)
    qdiag = _quantile_pair_diag(joined_df, qmap)
    qdiag.to_csv(out_dir / "quantile_pair_diagnostics.csv", index=False)

    scores_df, weights = _score_models(core_df, gate_df, tail_df)
    scores_df.to_csv(out_dir / "model_selection_scores.csv", index=False)
    (out_dir / "benchmark_score_weights.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")

    rec_df, rec_md = _recommend(scores_df)
    rec_df.to_csv(out_dir / "final_model_recommendation_table.csv", index=False)
    (out_dir / "final_model_recommendation_table.md").write_text(rec_md + "\n", encoding="utf-8")

    # compatibility file requested by validator style
    map_df.to_csv(out_dir / "benchmark_truth_mapping_audit.csv", index=False)

    manifest_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "manifests": [str(Path(m).resolve()) for m in args.manifests],
        "truth_override": str(Path(args.truth).resolve()) if args.truth else None,
        "split": args.split,
        "models": sorted(joined_df["model"].dropna().unique().tolist()),
        "targets": sorted(joined_df["target"].dropna().unique().tolist()),
        "quantiles_required": QUANTILES_REQUIRED,
        "quantile_pairs": _parse_quantile_pairs(args.quantile_pairs),
        "coverage_threshold": float(args.coverage_threshold),
        "plots_enabled": bool(args.plots),
        "output_dir": str(out_dir.resolve()),
    }
    (out_dir / "benchmark_manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "manifests_used": manifest_payload["manifests"],
        "truth_file_used": manifest_payload["truth_override"],
        "models_evaluated": manifest_payload["models"],
        "targets_evaluated": manifest_payload["targets"],
        "splits_evaluated": [args.split],
        "quantiles_evaluated": QUANTILES_REQUIRED,
        "output_paths": {
            "data_inventory": str((out_dir / "data_inventory.csv").resolve()),
            "forecast_metrics": str((out_dir / "forecast_metrics_probabilistic.csv").resolve()),
            "gate_metrics": str((out_dir / "gate_time_forecast_metrics.csv").resolve()),
            "tail_metrics": str((out_dir / "tail_performance_value_events.csv").resolve()),
            "joint_metrics": str((out_dir / "joint_value_event_diagnostics.csv").resolve()),
            "quantile_pair": str((out_dir / "quantile_pair_diagnostics.csv").resolve()),
            "scores": str((out_dir / "model_selection_scores.csv").resolve()),
            "recommendation": str((out_dir / "final_model_recommendation_table.csv").resolve()),
        },
        "warnings": warnings,
        "errors": [],
        "recommended_model_per_target": rec_df[["target", "recommended_model"]].to_dict(orient="records"),
        "missing_data_issues": inv_df.loc[inv_df["status"].ne("ok"), ["model", "target", "status"]].to_dict(orient="records"),
        "join_coverage_summary": {
            "min_pct": float(inv_df["join_coverage_pct"].min()),
            "mean_pct": float(inv_df["join_coverage_pct"].mean()),
        },
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run deterministic forecast benchmark for quantile prediction runs.")
    ap.add_argument("--manifests", nargs="+", required=True, help="One or more latest/run manifest paths")
    ap.add_argument("--truth", default="", help="Optional truth parquet override")
    ap.add_argument("--split", default="test", choices=["test", "val"], help="Split to evaluate")
    ap.add_argument("--out-dir", default="", help="Output directory")
    ap.add_argument("--models", default="", help="Optional model filter, e.g. xgb,tft,linear_torch")
    ap.add_argument("--targets", default="", help="Optional target filter over pred_* keys")
    ap.add_argument("--quantile-pairs", default="p50-p50,p70-p90")
    ap.add_argument("--coverage-threshold", type=float, default=0.999)
    ap.add_argument("--da-quantile-role", default="mid", choices=["low", "mid", "high"])
    ap.add_argument("--plots", dest="plots", action="store_true")
    ap.add_argument("--no-plots", dest="plots", action="store_false")
    ap.set_defaults(plots=False)
    ap.add_argument("--fail-on-missing", action="store_true", default=True)
    ap.add_argument("--overwrite", action="store_true", help="Alias for --clean-output")
    ap.add_argument("--clean-output", action="store_true")
    return ap.parse_args()


def _self_check() -> None:
    # lightweight synthetic checks
    t = pd.DataFrame({"timestamp_utc": pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"), "da_price": [1.0, 2.0, 3.0]})
    assert _resolve_truth_col(t, "pred_da_price") == "da_price"
    try:
        _resolve_truth_col(pd.DataFrame({"timestamp_utc": []}), "pred_da_price")
        raise AssertionError("Expected KeyError")
    except KeyError:
        pass
    try:
        _resolve_truth_col(pd.DataFrame({"timestamp_utc": [], "da_price": [], "target_da_price": []}), "pred_da_price")
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass

    y = np.array([0.0, 1.0, 2.0])
    yh = np.array([0.0, 1.0, 3.0])
    pb = _pinball(y, yh, 0.5)
    assert pb >= 0.0


if __name__ == "__main__":
    _self_check()
    args = parse_args()
    if args.overwrite:
        args.clean_output = True
    raise SystemExit(_run(args))
