#!/usr/bin/env python3
"""Build final RQ1 gate-specific actionable bucket benchmark outputs.

The script evaluates existing probabilistic forecast benchmark artifacts only.
It implements exactly two bucket families: general horizon buckets and the main
market-actionable DA/BCM/BEM buckets. Metrics are unweighted and model
comparisons are restricted to common valid rows.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_trading.evaluation.style import THESIS_PALETTE, apply_geo_style, get_model_color, thesis_titlecase
from energy_trading.evaluation.rq1_target_order import model_sort_key, ordered_unique, sort_target_frame, target_sort_key


BERLIN = ZoneInfo("Europe/Berlin")
QCOL_RE = re.compile(r"^p(\d{1,2})$", re.IGNORECASE)
DEFAULT_EVAL_ORIGIN_START_UTC = "2025-01-13T23:00:00Z"
DEFAULT_EVAL_ORIGIN_END_UTC = "2026-02-26T21:00:00Z"

MODEL_KEYS = {
    "tft": ("tft", "TFT"),
    "xgb": ("xgb", "XGB"),
    "xgboost": ("xgb", "XGB"),
    "linear": ("linear", "RLQR"),
    "rlqr": ("linear", "RLQR"),
}

TARGET_ALIASES = {
    "pred_da_price": "target_da_price",
    "target_da_price": "target_da_price",
    "pred_afrr_capacity_price_pos": "target_afrr_capacity_price_pos",
    "target_afrr_capacity_price_pos": "target_afrr_capacity_price_pos",
    "pred_afrr_capacity_price_neg": "target_afrr_capacity_price_neg",
    "target_afrr_capacity_price_neg": "target_afrr_capacity_price_neg",
    "pred_afrr_activation_price_pos": "target_afrr_activation_price_vwap_pos",
    "target_afrr_activation_price_pos": "target_afrr_activation_price_vwap_pos",
    "target_afrr_activation_price_vwap_pos": "target_afrr_activation_price_vwap_pos",
    "pred_afrr_activation_price_neg": "target_afrr_activation_price_vwap_neg",
    "target_afrr_activation_price_neg": "target_afrr_activation_price_vwap_neg",
    "target_afrr_activation_price_vwap_neg": "target_afrr_activation_price_vwap_neg",
    "pred_afrr_activation_rate_pos": "target_afrr_activation_rate_pos",
    "target_afrr_activation_rate_pos": "target_afrr_activation_rate_pos",
    "pred_afrr_activation_rate_neg": "target_afrr_activation_rate_neg",
    "target_afrr_activation_rate_neg": "target_afrr_activation_rate_neg",
}

TARGET_GROUPS = {
    "target_da_price": ("DA price", "DA price"),
    "target_afrr_capacity_price_pos": ("aFRR capacity price", "aFRR capacity price +"),
    "target_afrr_capacity_price_neg": ("aFRR capacity price", "aFRR capacity price -"),
    "target_afrr_activation_price_vwap_pos": ("aFRR activation price", "aFRR activation price +"),
    "target_afrr_activation_price_vwap_neg": ("aFRR activation price", "aFRR activation price -"),
    "target_afrr_activation_rate_pos": ("aFRR activation rate", "aFRR activation rate +"),
    "target_afrr_activation_rate_neg": ("aFRR activation rate", "aFRR activation rate -"),
}

CAPACITY_TARGETS = {"target_afrr_capacity_price_pos", "target_afrr_capacity_price_neg"}
ACTIVATION_TARGETS = {
    "target_afrr_activation_price_vwap_pos",
    "target_afrr_activation_price_vwap_neg",
    "target_afrr_activation_rate_pos",
    "target_afrr_activation_rate_neg",
}
ALL_CANONICAL_TARGETS = {"target_da_price", *CAPACITY_TARGETS, *ACTIVATION_TARGETS}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str


@dataclass(frozen=True)
class BucketSpec:
    bucket: str
    family: str
    description: str
    target_filter: set[str] | None
    expected_leads: tuple[int, ...] | None = None
    expected_leads_alt: tuple[int, ...] | None = None


BUCKETS = [
    BucketSpec("full_h1_48", "horizon", "All forecast horizons h1-h48.", None, tuple(range(1, 49))),
    BucketSpec("short_h1_8", "horizon", "Short forecast horizon h1-h8.", None, tuple(range(1, 9))),
    BucketSpec("medium_h9_16", "horizon", "Medium forecast horizon h9-h16.", None, tuple(range(9, 17))),
    BucketSpec("long_h17_48", "horizon", "Long forecast horizon h17-h48.", None, tuple(range(17, 49))),
    BucketSpec(
        "actionable_da_dplus1_11",
        "actionable",
        "DA forecast snapshot at 11:00 Europe/Berlin for next local delivery day 00:00-23:00.",
        {"target_da_price"},
        tuple(range(13, 37)),
    ),
    BucketSpec(
        "actionable_bcm_dplus1_08",
        "actionable",
        "BCM aFRR capacity forecast snapshot at 08:00 Europe/Berlin for next local delivery day 00:00-23:00.",
        CAPACITY_TARGETS,
        tuple(range(16, 40)),
        (16, 20, 24, 28, 32, 36),
    ),
    BucketSpec(
        "actionable_bem_short_h1_8",
        "actionable",
        "Short-term BEM activation relevance bucket h1-h8.",
        ACTIVATION_TARGETS,
        tuple(range(1, 9)),
    ),
]

BUCKET_LABELS = {
    "full_h1_48": "Full horizon (h1--h48)",
    "short_h1_8": "Short horizon (h1--h8)",
    "medium_h9_16": "Medium horizon (h9--h16)",
    "long_h17_48": "Long horizon (h17--h48)",
}


def _bucket_label(bucket: Any) -> str:
    return BUCKET_LABELS.get(str(bucket), str(bucket))


def _parse_models(raw: str) -> list[ModelSpec]:
    out: list[ModelSpec] = []
    seen: set[str] = set()
    for item in str(raw).split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in MODEL_KEYS:
            raise ValueError(f"Unknown model key {item!r}. Supported: {', '.join(sorted(MODEL_KEYS))}")
        canonical, label = MODEL_KEYS[key]
        if canonical not in seen:
            seen.add(canonical)
            out.append(ModelSpec(canonical, label))
    if not out:
        raise ValueError("At least one model is required.")
    return out


def canonical_target(target: Any) -> str:
    raw = str(target)
    return TARGET_ALIASES.get(raw, raw)


def _target_info(target: str) -> tuple[str, str]:
    return TARGET_GROUPS.get(canonical_target(target), ("Other", str(target).replace("_", " ")))


def _qcol(q: float) -> str:
    return f"p{int(round(float(q) * 100)):02d}"


def _quantile_cols(df: pd.DataFrame) -> dict[float, str]:
    out: dict[float, str] = {}
    for col in df.columns:
        m = QCOL_RE.match(str(col).lower())
        if m:
            q = int(m.group(1)) / 100.0
            if 0.0 < q < 1.0:
                out[q] = str(col)
    return out


def _parse_joined_name(path: Path) -> tuple[str, str, str] | None:
    parts = path.stem.split("__", 2)
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def _discover_benchmark_dir(benchmark_root: Path, benchmark_dir: Path | None) -> Path:
    if benchmark_dir is not None:
        out = benchmark_dir.resolve()
    elif (benchmark_root / "diagnostics" / "joined_predictions").exists():
        out = benchmark_root.resolve()
    else:
        if not benchmark_root.exists():
            raise FileNotFoundError(
                f"No benchmark directory found because benchmark root does not exist: {benchmark_root}. "
                "Run scripts/run_forecast_benchmark.py with --save-joined-predictions first."
            )
        candidates = sorted(
            [p.resolve() for p in benchmark_root.iterdir() if (p / "diagnostics" / "joined_predictions").exists()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No benchmark directory with diagnostics/joined_predictions found under {benchmark_root}. "
                "Run scripts/run_forecast_benchmark.py with --save-joined-predictions first."
            )
        if len(candidates) > 1:
            raise ValueError(
                "Multiple benchmark directories were found. Pass --benchmark-dir explicitly. "
                f"Candidates: {', '.join(str(p) for p in candidates[:5])}"
            )
        out = candidates[0]
    joined = out / "diagnostics" / "joined_predictions"
    if not joined.exists():
        raise FileNotFoundError(
            f"Missing joined predictions directory: {joined}. "
            "Run scripts/run_forecast_benchmark.py with --save-joined-predictions first."
        )
    return out


def _read_joined(path: Path, *, source_target: str, derive_forecast_time: bool) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    df = pd.read_parquet(path).copy()
    missing = {"target_time_utc", "lead_time_h", "y_true"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df["target_time_utc"] = pd.to_datetime(df["target_time_utc"], utc=True, errors="coerce")
    df["lead_time_h"] = pd.to_numeric(df["lead_time_h"], errors="coerce")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    if "forecast_time_utc" in df.columns:
        df["forecast_time_utc"] = pd.to_datetime(df["forecast_time_utc"], utc=True, errors="coerce")
    elif "snapshot_time_utc" in df.columns:
        df["forecast_time_utc"] = pd.to_datetime(df["snapshot_time_utc"], utc=True, errors="coerce")
    elif derive_forecast_time:
        df["forecast_time_utc"] = df["target_time_utc"] - pd.to_timedelta(df["lead_time_h"], unit="h")
        warnings.append(
            {
                "split": None,
                "target": canonical_target(source_target),
                "bucket": None,
                "severity": "warning",
                "message": "forecast_time_utc was derived as target_time_utc - lead_time_h because joined predictions do not store forecast_time_utc.",
            }
        )
    else:
        raise ValueError(
            f"{path} is missing forecast_time_utc/snapshot_time_utc. "
            "Actionable DA/BCM gate buckets require forecast timestamps; rerun the forecast benchmark "
            "with joined predictions that preserve forecast_time_utc, or explicitly pass "
            "--derive-forecast-time-from-lead for a documented legacy fallback."
        )
    if "p50" not in df.columns and "predicted_value" in df.columns:
        df["p50"] = pd.to_numeric(df["predicted_value"], errors="coerce")
    if "p50" not in df.columns:
        raise ValueError(f"{path} must contain p50 or predicted_value.")
    df["target"] = canonical_target(source_target)
    return df, warnings


def _parse_utc_bound(raw: str | None) -> pd.Timestamp | None:
    if raw is None or str(raw).strip() == "":
        return None
    return pd.to_datetime(raw, utc=True)


def _apply_forecast_origin_window(df: pd.DataFrame, *, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    if start is None and end is None:
        return df
    origin = pd.to_datetime(df["forecast_time_utc"], utc=True, errors="coerce")
    mask = origin.notna()
    if start is not None:
        mask &= origin.ge(start)
    if end is not None:
        mask &= origin.le(end)
    return df.loc[mask].copy()


def bucket_mask(df: pd.DataFrame, bucket: str) -> pd.Series:
    if bucket not in {b.bucket for b in BUCKETS}:
        raise ValueError(f"Unknown bucket: {bucket}")
    d = df.copy()
    if "target" not in d.columns:
        raise KeyError("Bucket evaluation requires a target column.")
    target = d["target"].map(canonical_target)
    lead = pd.to_numeric(d.get("lead_time_h"), errors="coerce")

    if bucket == "full_h1_48":
        return lead.between(1, 48, inclusive="both")
    if bucket == "short_h1_8":
        return lead.between(1, 8, inclusive="both")
    if bucket == "medium_h9_16":
        return lead.between(9, 16, inclusive="both")
    if bucket == "long_h17_48":
        return lead.between(17, 48, inclusive="both")
    if bucket == "actionable_bem_short_h1_8":
        return target.isin(ACTIVATION_TARGETS) & lead.between(1, 8, inclusive="both")

    required = {"forecast_time_utc", "target_time_utc"}
    missing = required - set(d.columns)
    if missing:
        raise KeyError(f"Actionable DA/BCM bucket evaluation requires timestamp columns: {sorted(missing)}")

    forecast = pd.to_datetime(d["forecast_time_utc"], utc=True, errors="coerce").dt.tz_convert(BERLIN)
    target_time = pd.to_datetime(d["target_time_utc"], utc=True, errors="coerce").dt.tz_convert(BERLIN)
    same_next_day = target_time.dt.date == (forecast + pd.Timedelta(days=1)).dt.date
    delivery_day = target_time.dt.hour.between(0, 23, inclusive="both")
    on_hour = forecast.dt.minute.eq(0) & forecast.dt.second.eq(0)
    if bucket == "actionable_da_dplus1_11":
        return target.eq("target_da_price") & forecast.dt.hour.eq(11) & on_hour & same_next_day & delivery_day
    if bucket == "actionable_bcm_dplus1_08":
        return target.isin(CAPACITY_TARGETS) & forecast.dt.hour.eq(8) & on_hour & same_next_day & delivery_day
    raise ValueError(f"Unhandled bucket: {bucket}")


def _valid_frame(df: pd.DataFrame, qcols: dict[float, str]) -> pd.DataFrame:
    cols = list(dict.fromkeys(["forecast_time_utc", "target_time_utc", "lead_time_h", "y_true", "p50", *[qcols[q] for q in sorted(qcols)]]))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for valid frame: {missing}")
    out = df[cols].copy()
    for col in ["y_true", "p50", *[qcols[q] for q in sorted(qcols)]]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    mask = out[["forecast_time_utc", "target_time_utc", "lead_time_h"]].notna().all(axis=1)
    for col in ["y_true", "p50", *[qcols[q] for q in sorted(qcols)]]:
        mask &= np.isfinite(out[col].to_numpy(dtype=float))
    return out.loc[mask].copy()


def _key_tuples(df: pd.DataFrame) -> set[tuple[Any, ...]]:
    return set(map(tuple, df[["forecast_time_utc", "target_time_utc", "lead_time_h"]].itertuples(index=False, name=None)))


def _pinball_values(y: np.ndarray, pred: np.ndarray, q: float) -> np.ndarray:
    err = y - pred
    return np.maximum(q * err, (q - 1.0) * err)


def _lead_summary(leads: pd.Series) -> str:
    vals = sorted({int(x) for x in pd.to_numeric(leads, errors="coerce").dropna().tolist()})
    if not vals:
        return ""
    ranges: list[str] = []
    start = prev = vals[0]
    for val in vals[1:]:
        if val == prev + 1:
            prev = val
            continue
        ranges.append(f"h{start}" if start == prev else f"h{start}-h{prev}")
        start = prev = val
    ranges.append(f"h{start}" if start == prev else f"h{start}-h{prev}")
    return ",".join(ranges)


def _metrics(df: pd.DataFrame, qcols: dict[float, str]) -> dict[str, Any]:
    y = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    p50 = pd.to_numeric(df["p50"], errors="coerce").to_numpy(dtype=float)
    pinball_parts = [
        _pinball_values(y, pd.to_numeric(df[qcols[q]], errors="coerce").to_numpy(dtype=float), q)
        for q in sorted(qcols)
    ]
    out: dict[str, Any] = {
        "mean_pinball_loss": float(np.mean(np.vstack(pinball_parts))) if pinball_parts else float("nan"),
        "mae_p50": float(np.mean(np.abs(p50 - y))),
        "rmse_p50": float(np.sqrt(np.mean((p50 - y) ** 2))),
        "bias_p50": float(np.mean(p50 - y)),
        "n_obs": int(len(df)),
        "n_forecast_timestamps": int(df["forecast_time_utc"].nunique()),
        "n_target_timestamps": int(df["target_time_utc"].nunique()),
        "observed_lead_min": float(pd.to_numeric(df["lead_time_h"], errors="coerce").min()),
        "observed_lead_max": float(pd.to_numeric(df["lead_time_h"], errors="coerce").max()),
        "observed_leads": _lead_summary(df["lead_time_h"]),
    }
    for q in sorted(qcols):
        pred = pd.to_numeric(df[qcols[q]], errors="coerce").to_numpy(dtype=float)
        out[f"coverage_{_qcol(q)}"] = float(np.mean(y <= pred))
    if 0.10 in qcols and 0.90 in qcols:
        lo = pd.to_numeric(df[qcols[0.10]], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(df[qcols[0.90]], errors="coerce").to_numpy(dtype=float)
        out["coverage_p10_p90"] = float(np.mean((lo <= y) & (y <= hi)))
        out["interval_width_p10_p90_mean"] = float(np.mean(hi - lo))
    return out


def _bucket_definitions_frame() -> pd.DataFrame:
    rows = []
    for spec in BUCKETS:
        rows.append(
            {
                "bucket": spec.bucket,
                "family": spec.family,
                "description": spec.description,
                "target_filter": ",".join(sorted(spec.target_filter)) if spec.target_filter else "all",
                "expected_leads": _lead_summary(pd.Series(spec.expected_leads)) if spec.expected_leads else "",
                "expected_leads_alt": _lead_summary(pd.Series(spec.expected_leads_alt)) if spec.expected_leads_alt else "",
            }
        )
    return pd.DataFrame(rows)


def build_gate_bucket_outputs(
    *,
    benchmark_dir: Path,
    models: list[ModelSpec],
    splits: list[str],
    derive_forecast_time: bool,
    eval_origin_start: pd.Timestamp | None,
    eval_origin_end: pd.Timestamp | None,
) -> dict[str, pd.DataFrame]:
    joined_dir = benchmark_dir / "diagnostics" / "joined_predictions"
    files: dict[tuple[str, str, str], Path] = {}
    for path in sorted(joined_dir.glob("*.parquet")):
        parsed = _parse_joined_name(path)
        if parsed is not None:
            files[parsed] = path
    targets = sorted({target for _, split, target in files if split in set(splits) and canonical_target(target) in ALL_CANONICAL_TARGETS}, key=target_sort_key)
    if not targets:
        raise FileNotFoundError(f"No supported joined prediction parquet files for splits={splits} in {joined_dir}.")

    metric_rows: list[dict[str, Any]] = []
    row_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []

    for split in splits:
        for source_target in targets:
            canonical = canonical_target(source_target)
            target_group, target_label = _target_info(canonical)
            loaded: dict[str, pd.DataFrame] = {}
            qmaps: dict[str, dict[float, str]] = {}
            for model in models:
                path = files.get((model.key, split, source_target))
                if path is None:
                    raise FileNotFoundError(f"Missing joined predictions for model={model.key}, split={split}, target={source_target}.")
                df, read_warnings = _read_joined(path, source_target=source_target, derive_forecast_time=derive_forecast_time)
                df = _apply_forecast_origin_window(df, start=eval_origin_start, end=eval_origin_end)
                for warning in read_warnings:
                    warning_rows.append({**warning, "split": split})
                loaded[model.key] = df
                qmaps[model.key] = _quantile_cols(df)
                if 0.50 not in qmaps[model.key]:
                    raise ValueError(f"Missing p50 for model={model.key}, split={split}, target={source_target}.")

            for spec in BUCKETS:
                if spec.target_filter is not None and canonical not in spec.target_filter:
                    continue
                selected: dict[str, pd.DataFrame] = {}
                for model in models:
                    mask = bucket_mask(loaded[model.key], spec.bucket)
                    selected[model.key] = loaded[model.key].loc[mask].copy()
                observed_all = pd.concat(selected.values(), ignore_index=True) if selected else pd.DataFrame()
                if observed_all.empty:
                    warning_rows.append(
                        {
                            "split": split,
                            "target": canonical,
                            "bucket": spec.bucket,
                            "severity": "warning",
                            "message": "Bucket selected zero rows.",
                        }
                    )
                    continue
                observed_leads = sorted({int(x) for x in pd.to_numeric(observed_all["lead_time_h"], errors="coerce").dropna().unique()})
                lead_rows.append(
                    {
                        "split": split,
                        "bucket": spec.bucket,
                        "target": canonical,
                        "target_group": target_group,
                        "observed_lead_min": min(observed_leads) if observed_leads else np.nan,
                        "observed_lead_max": max(observed_leads) if observed_leads else np.nan,
                        "observed_leads": _lead_summary(pd.Series(observed_leads)),
                        "n_rows_before_common_intersection": int(sum(len(x) for x in selected.values())),
                    }
                )
                if spec.expected_leads:
                    obs_set = set(observed_leads)
                    exp = set(spec.expected_leads)
                    alt = set(spec.expected_leads_alt or ())
                    if obs_set and obs_set != exp and not (alt and obs_set == alt):
                        warning_rows.append(
                            {
                                "split": split,
                                "target": canonical,
                                "bucket": spec.bucket,
                                "severity": "warning",
                                "message": f"Observed leads {_lead_summary(pd.Series(observed_leads))} differ from expected {_lead_summary(pd.Series(spec.expected_leads))}.",
                            }
                        )

                common_qs = set.intersection(*(set(qmaps[m.key]) for m in models))
                if not common_qs:
                    raise ValueError(f"No common quantile grid for split={split}, target={canonical}, bucket={spec.bucket}.")
                common_qcols = {m.key: {q: qmaps[m.key][q] for q in sorted(common_qs)} for m in models}
                valid = {m.key: _valid_frame(selected[m.key], common_qcols[m.key]) for m in models}
                common_keys = set.intersection(*(_key_tuples(valid[m.key]) for m in models))
                if not common_keys:
                    warning_rows.append(
                        {
                            "split": split,
                            "target": canonical,
                            "bucket": spec.bucket,
                            "severity": "warning",
                            "message": "Common valid row intersection is empty.",
                        }
                    )
                    continue
                key_df = pd.DataFrame(list(common_keys), columns=["forecast_time_utc", "target_time_utc", "lead_time_h"])
                quantiles_used = ",".join(_qcol(q) for q in sorted(common_qs))
                for model in models:
                    original_rows = int(len(selected[model.key]))
                    valid_rows = int(len(valid[model.key]))
                    retained_rows = int(len(common_keys))
                    dropped_rows = int(valid_rows - retained_rows)
                    row_rows.append(
                        {
                            "split": split,
                            "bucket": spec.bucket,
                            "target": canonical,
                            "target_group": target_group,
                            "model": model.key,
                            "model_label": model.label,
                            "original_rows": original_rows,
                            "valid_rows": valid_rows,
                            "retained_common_rows": retained_rows,
                            "dropped_rows": dropped_rows,
                            "retained_share": retained_rows / valid_rows if valid_rows else float("nan"),
                            "quantiles_available": ",".join(_qcol(q) for q in sorted(qmaps[model.key])),
                            "quantiles_used": quantiles_used,
                            "row_intersection_key": "split,target,forecast_time_utc,target_time_utc,lead_time_h",
                            "eval_origin_start_utc": eval_origin_start.isoformat() if eval_origin_start is not None else "",
                            "eval_origin_end_utc": eval_origin_end.isoformat() if eval_origin_end is not None else "",
                        }
                    )
                    eval_df = valid[model.key].merge(key_df, on=["forecast_time_utc", "target_time_utc", "lead_time_h"], how="inner")
                    metric_rows.append(
                        {
                            "model": model.key,
                            "model_label": model.label,
                            "split": split,
                            "bucket": spec.bucket,
                            "bucket_family": spec.family,
                            "target": canonical,
                            "target_label": target_label,
                            "target_group": target_group,
                            "quantiles_used": quantiles_used,
                            **_metrics(eval_df, common_qcols[model.key]),
                        }
                    )

    return {
        "metrics": sort_target_frame(pd.DataFrame(metric_rows), target_col="target", extra_cols=["split", "bucket", "model_label"]),
        "row_counts": sort_target_frame(pd.DataFrame(row_rows), target_col="target", extra_cols=["split", "bucket", "model_label"]),
        "definitions": _bucket_definitions_frame(),
        "observed_leads": sort_target_frame(pd.DataFrame(lead_rows), target_col="target", extra_cols=["split", "bucket"]),
        "warnings": pd.DataFrame(warning_rows, columns=["split", "target", "bucket", "severity", "message"]),
    }


def _latex_escape(value: Any) -> str:
    s = str(value)
    minus_token = "@@RQ1MINUS@@"
    for label in ["aFRR capacity price", "aFRR activation price", "aFRR activation rate"]:
        s = s.replace(f"{label} -", f"{label} {minus_token}")
        s = s.replace(f"{label} \u2212", f"{label} {minus_token}")
    for old, new in {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }.items():
        s = s.replace(old, new)
    return s.replace(minus_token, "$-$")


def _fmt(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "-"
    return f"{x:.4f}" if np.isfinite(x) else "-"


def _main_table(metrics: pd.DataFrame, *, split: str) -> pd.DataFrame:
    d = metrics.loc[metrics["split"] == split].copy()
    if d.empty:
        return pd.DataFrame()
    d = sort_target_frame(d, target_col="target", extra_cols=["bucket", "model_label"])
    rows: list[dict[str, Any]] = []
    for (bucket, target_group), part in d.groupby(["bucket", "target_group"], sort=False):
        vals = {
            str(label): float(group["mean_pinball_loss"].mean())
            for label, group in part.groupby("model_label")
            if pd.to_numeric(group["mean_pinball_loss"], errors="coerce").notna().any()
        }
        best = min(vals, key=vals.get) if vals else ""
        rows.append(
            {
                "bucket": bucket,
                "target_group": target_group,
                "RLQR": vals.get("RLQR", np.nan),
                "XGB": vals.get("XGB", np.nan),
                "TFT": vals.get("TFT", np.nan),
                "best_model": best,
                "n_obs": int(part.groupby("model_label")["n_obs"].sum().min()) if not part.empty else 0,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_target_order"] = out["target_group"].map(lambda x: target_sort_key(str(x))[0])
    out["_bucket_order"] = out["bucket"].map({spec.bucket: i for i, spec in enumerate(BUCKETS)}).fillna(99)
    return out.sort_values(["_target_order", "_bucket_order"]).drop(columns=["_target_order", "_bucket_order"]).reset_index(drop=True)


def write_latex_table(metrics: pd.DataFrame, *, out_dir: Path, split: str) -> Path | None:
    table = _main_table(metrics, split=split)
    if table.empty:
        return None
    headers = ["Bucket", "Target group", "RLQR", "XGB", "TFT", "Best model", "N"]
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \begin{tabular}{@{}llrrrrr@{}}",
        r"        \toprule",
        "        " + " & ".join(r"\textbf{" + _latex_escape(h) + "}" for h in headers) + r" \\",
        r"        \midrule",
    ]
    for _, row in table.iterrows():
        vals = [
            r"\textbf{" + _latex_escape(_bucket_label(row["bucket"])) + "}",
            _latex_escape(row["target_group"]),
            _fmt(row["RLQR"]),
            _fmt(row["XGB"]),
            _fmt(row["TFT"]),
            _latex_escape(row["best_model"]),
            str(int(row["n_obs"])),
        ]
        lines.append("        " + " & ".join(vals) + r" \\")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \caption{Gate-specific and horizon-bucket mean pinball loss on the test split. Values are unweighted and computed on common valid forecast rows across models.}",
            r"    \label{tab:gate_bucket_metrics_test}",
            r"\end{table}",
            "",
        ]
    )
    path = out_dir / "latex" / f"gate_bucket_metrics_{split}.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _plot_metric(metrics: pd.DataFrame, *, out_dir: Path, split: str, metric: str, filename: str, ylabel: str) -> Path | None:
    import matplotlib.pyplot as plt

    d = metrics.loc[metrics["split"] == split].copy()
    if d.empty or metric not in d.columns:
        return None
    apply_geo_style()
    groups = ordered_unique(d["target_group"].dropna().unique(), group=True)
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, max(3.2, 2.7 * len(groups))), sharex=False)
    if len(groups) == 1:
        axes = [axes]
    for ax, group in zip(axes, groups):
        panel = d[d["target_group"] == group].copy()
        agg = panel.groupby(["bucket", "model", "model_label"], as_index=False, sort=False).agg(value=(metric, "mean"), n_obs=("n_obs", "sum"))
        bucket_order = {spec.bucket: i for i, spec in enumerate(BUCKETS)}
        buckets = sorted(agg["bucket"].unique(), key=lambda b: bucket_order.get(str(b), 99))
        x = np.arange(len(buckets), dtype=float)
        width = 0.24
        models = sorted(agg["model"].dropna().unique(), key=model_sort_key)
        for idx, model in enumerate(models):
            mg = agg[agg["model"].eq(model)]
            vals = [float(mg.loc[mg["bucket"] == b, "value"].mean()) for b in buckets]
            ax.bar(x + (idx - 1) * width, vals, width=width, label=str(mg["model_label"].iloc[0]), color=get_model_color(str(model)))
        for i, bucket in enumerate(buckets):
            n = int(agg.loc[agg["bucket"] == bucket, "n_obs"].max()) if not agg.loc[agg["bucket"] == bucket].empty else 0
            ax.text(i, ax.get_ylim()[1] * 0.98, f"n={n}", ha="center", va="top", fontsize=7)
        ax.set_title(thesis_titlecase(group))
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([_bucket_label(bucket) for bucket in buckets], rotation=25, ha="right")
        ax.legend(ncol=3, loc="upper left")
    fig.suptitle(thesis_titlecase(f"{ylabel} by gate/actionable bucket ({split}; lower is better except coverage)"), y=1.01)
    fig.tight_layout()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_relative_pinball(metrics: pd.DataFrame, *, out_dir: Path, split: str, filename: str) -> Path | None:
    import matplotlib.pyplot as plt

    d = metrics.loc[metrics["split"] == split].copy()
    required = {"target_group", "bucket", "model", "model_label", "mean_pinball_loss", "n_obs"}
    if d.empty or not required.issubset(d.columns):
        return None
    apply_geo_style()
    groups = ordered_unique(d["target_group"].dropna().unique(), group=True)
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, max(3.2, 2.7 * len(groups))), sharex=False)
    if len(groups) == 1:
        axes = [axes]
    bucket_order = {spec.bucket: i for i, spec in enumerate(BUCKETS)}
    for ax, group in zip(axes, groups):
        panel = d[d["target_group"].eq(group)].copy()
        agg = panel.groupby(["bucket", "model_label"], as_index=False, sort=False).agg(
            value=("mean_pinball_loss", "mean"),
            n_obs=("n_obs", "sum"),
        )
        pivot = agg.pivot_table(index="bucket", columns="model_label", values="value", aggfunc="mean").reset_index()
        counts = agg.groupby("bucket", as_index=False)["n_obs"].max()
        pivot = pivot.merge(counts, on="bucket", how="left")
        if "RLQR" not in pivot.columns:
            continue
        pivot = pivot[pd.to_numeric(pivot["RLQR"], errors="coerce").notna() & pd.to_numeric(pivot["RLQR"], errors="coerce").ne(0)].copy()
        if pivot.empty:
            continue
        buckets = sorted(pivot["bucket"].dropna().unique(), key=lambda b: bucket_order.get(str(b), 99))
        x = np.arange(len(buckets), dtype=float)
        width = 0.32
        ax.axhline(1.0, color=get_model_color("linear"), linewidth=1.4, linestyle=":", label="RLQR")
        for idx, (label, model_key) in enumerate([("XGB", "xgb"), ("TFT", "tft")]):
            if label not in pivot.columns:
                continue
            vals: list[float] = []
            for bucket in buckets:
                row = pivot[pivot["bucket"].eq(bucket)]
                if row.empty:
                    vals.append(float("nan"))
                    continue
                denom = float(row["RLQR"].iloc[0])
                value = float(row[label].iloc[0]) if pd.notna(row[label].iloc[0]) else float("nan")
                vals.append(value / denom if np.isfinite(denom) and abs(denom) > 1e-12 else float("nan"))
            ax.bar(x + (idx - 0.5) * width, vals, width=width, label=label, color=get_model_color(model_key))
        for i, bucket in enumerate(buckets):
            n = int(pivot.loc[pivot["bucket"] == bucket, "n_obs"].max()) if not pivot.loc[pivot["bucket"] == bucket].empty else 0
            ax.text(i, ax.get_ylim()[1] * 0.98, f"n={n}", ha="center", va="top", fontsize=7)
        ax.set_title(thesis_titlecase(group))
        ax.set_ylabel("Mean pinball loss relative to RLQR")
        ax.set_xticks(x)
        ax.set_xticklabels([_bucket_label(bucket) for bucket in buckets], rotation=25, ha="right")
        ax.legend(ncol=3, loc="upper left")
    fig.suptitle(thesis_titlecase("Gate-specific relative mean pinball loss (RLQR = 1)"), y=1.01)
    fig.tight_layout()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_observed_leads(observed: pd.DataFrame, *, out_dir: Path, split: str) -> Path | None:
    import matplotlib.pyplot as plt

    d = observed.loc[observed["split"] == split].copy()
    if d.empty:
        return None
    apply_geo_style()
    d["label"] = d["bucket"].map(_bucket_label) + " | " + d["target_group"]
    d = sort_target_frame(d, target_col="target", extra_cols=["bucket"])
    y = np.arange(len(d), dtype=float)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.28 * len(d))))
    mins = pd.to_numeric(d["observed_lead_min"], errors="coerce")
    maxs = pd.to_numeric(d["observed_lead_max"], errors="coerce")
    ax.hlines(y, mins, maxs, color=THESIS_PALETTE["primary"], linewidth=2.0)
    ax.scatter(mins, y, color=THESIS_PALETTE["primary"], s=20)
    ax.scatter(maxs, y, color=THESIS_PALETTE["tertiary"], s=20)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlabel("Observed lead hour")
    ax.set_title(thesis_titlecase("Observed lead ranges by gate/actionable bucket"))
    ax.set_xlim(0, 49)
    fig.tight_layout()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "gate_bucket_observed_leads.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_outputs(outputs: dict[str, pd.DataFrame], *, out_dir: Path, split: str, structured_out_dir: Path | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latex").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    csvs = [
        ("gate_bucket_metrics.csv", outputs["metrics"]),
        (f"gate_bucket_metrics_{split}.csv", outputs["metrics"].loc[outputs["metrics"]["split"] == split]),
        ("gate_bucket_row_counts.csv", outputs["row_counts"]),
        ("gate_bucket_definitions.csv", outputs["definitions"]),
        ("gate_bucket_observed_leads.csv", outputs["observed_leads"]),
        ("gate_bucket_warnings.csv", outputs["warnings"]),
    ]
    for name, df in csvs:
        path = out_dir / name
        df.to_csv(path, index=False)
        paths.append(path)
    tex = write_latex_table(outputs["metrics"], out_dir=out_dir, split=split)
    if tex is not None:
        paths.append(tex)
    for p in [
        _plot_relative_pinball(outputs["metrics"], out_dir=out_dir, split=split, filename="gate_bucket_pinball_by_target_group.png"),
        _plot_metric(outputs["metrics"], out_dir=out_dir, split=split, metric="mae_p50", filename="gate_bucket_mae_p50_by_target_group.png", ylabel="MAE p50"),
        _plot_metric(outputs["metrics"], out_dir=out_dir, split=split, metric="coverage_p10_p90", filename="gate_bucket_coverage_p10_p90_by_target_group.png", ylabel="p10-p90 empirical coverage"),
        _plot_observed_leads(outputs["observed_leads"], out_dir=out_dir, split=split),
    ]:
        if p is not None:
            paths.append(p)
    manifest = {
        "description": "RQ1 gate-specific actionable bucket benchmark outputs.",
        "split": split,
        "buckets": [b.bucket for b in BUCKETS],
        "metrics": ["mean_pinball_loss", "mae_p50", "rmse_p50", "bias_p50", "coverage_p10_p90", "interval_width_p10_p90_mean"],
        "row_intersection_key": "split,target,forecast_time_utc,target_time_utc,lead_time_h",
        "outputs": [str(p) for p in paths],
    }
    manifest_path = out_dir / "gate_bucket_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths.append(manifest_path)
    if structured_out_dir is not None:
        paths.extend(_mirror_structured(paths, root_out_dir=out_dir, structured_out_dir=structured_out_dir))
    return paths


def _mirror_structured(paths: list[Path], *, root_out_dir: Path, structured_out_dir: Path) -> list[Path]:
    mirrored: list[Path] = []
    for src in paths:
        if not src.exists():
            continue
        try:
            rel = src.relative_to(root_out_dir)
        except ValueError:
            continue
        dst = structured_out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        mirrored.append(dst)
    return mirrored


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build final RQ1 gate-specific actionable bucket benchmark outputs.")
    p.add_argument("--benchmark-root", default="artifacts")
    p.add_argument("--benchmark-dir", default=None)
    p.add_argument("--out-dir", default="artifacts/rq1_ml_model_benchmark/_raw_outputs/shared")
    p.add_argument("--structured-out-dir", default="artifacts/rq1_ml_model_benchmark/_raw_outputs/4_1_4_gate_specific")
    p.add_argument("--split", default="test", help="Main thesis/reporting split. Defaults to test.")
    p.add_argument(
        "--splits",
        default="test",
        help="Comma-separated splits to load/export. Defaults to test only; --split selects the main reported split.",
    )
    p.add_argument("--models", default="tft,xgboost,linear", help="Models to compare. Defaults to all RQ1 models: TFT, XGB and RLQR.")
    p.add_argument("--derive-forecast-time-from-lead", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-origin-start", default=DEFAULT_EVAL_ORIGIN_START_UTC, help="Inclusive forecast-origin lower bound for final RQ1 evaluation. Empty string disables the lower bound.")
    p.add_argument("--eval-origin-end", default=DEFAULT_EVAL_ORIGIN_END_UTC, help="Inclusive forecast-origin upper bound for final RQ1 evaluation. Empty string disables the upper bound.")
    p.add_argument("--no-structured-copy", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    models = _parse_models(args.models)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    if args.split not in splits:
        splits.append(args.split)
    benchmark_dir = _discover_benchmark_dir(
        Path(args.benchmark_root),
        Path(args.benchmark_dir) if args.benchmark_dir else None,
    )
    outputs = build_gate_bucket_outputs(
        benchmark_dir=benchmark_dir,
        models=models,
        splits=splits,
        derive_forecast_time=bool(args.derive_forecast_time_from_lead),
        eval_origin_start=_parse_utc_bound(args.eval_origin_start),
        eval_origin_end=_parse_utc_bound(args.eval_origin_end),
    )
    structured_out_dir = None if args.no_structured_copy else Path(args.structured_out_dir)
    paths = write_outputs(outputs, out_dir=Path(args.out_dir), split=args.split, structured_out_dir=structured_out_dir)
    print("[OK] Built RQ1 gate-specific bucket outputs.")
    print(f"[OK] benchmark_dir={benchmark_dir}")
    for path in paths:
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
