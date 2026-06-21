#!/usr/bin/env python3
"""Build RQ1 4.1.1 full unweighted forecast metrics.

This script is intentionally limited to the general full-row benchmark:
mean pinball loss is the main-text metric, while p50 MAE/RMSE/bias are
exported as diagnostics. It does not compute interval calibration, gate,
lead-hour, tail/spike, simulation, HPO, or training outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_trading.evaluation.rq1_target_order import (MODEL_LABELS,
                                                        model_sort_key,
                                                        sort_model_frame,
                                                        sort_target_frame,
                                                        target_sort_key)
from energy_trading.visualization.style import (THESIS_PALETTE,
                                                apply_geo_style,
                                                get_model_color,
                                                thesis_titlecase)

MODEL_KEYS = {
    "tft": ("tft", "TFT"),
    "xgb": ("xgb", "XGB"),
    "xgboost": ("xgb", "XGB"),
    "linear": ("linear", "RLQR"),
    "rlqr": ("linear", "RLQR"),
}

TARGET_GROUPS = {
    "pred_da_price": ("DA", "DA price"),
    "target_da_price": ("DA", "DA price"),
    "pred_afrr_capacity_price_pos": ("aFRR capacity", "aFRR capacity price +"),
    "target_afrr_capacity_price_pos": ("aFRR capacity", "aFRR capacity price +"),
    "pred_afrr_capacity_price_neg": ("aFRR capacity", "aFRR capacity price -"),
    "target_afrr_capacity_price_neg": ("aFRR capacity", "aFRR capacity price -"),
    "pred_afrr_activation_price_pos": ("aFRR activation price", "aFRR activation price +"),
    "target_afrr_activation_price_pos": ("aFRR activation price", "aFRR activation price +"),
    "pred_afrr_activation_price_neg": ("aFRR activation price", "aFRR activation price -"),
    "target_afrr_activation_price_neg": ("aFRR activation price", "aFRR activation price -"),
    "pred_afrr_activation_rate_pos": ("aFRR activation rate", "aFRR activation rate +"),
    "target_afrr_activation_rate_pos": ("aFRR activation rate", "aFRR activation rate +"),
    "pred_afrr_activation_rate_neg": ("aFRR activation rate", "aFRR activation rate -"),
    "target_afrr_activation_rate_neg": ("aFRR activation rate", "aFRR activation rate -"),
}

METRICS = ["mean_pinball_loss", "mae_p50", "rmse_p50", "bias_p50"]
LOWER_IS_BETTER = {
    "mean_pinball_loss": True,
    "mae_p50": True,
    "rmse_p50": True,
    "bias_p50": False,
}
METRIC_LABELS = {
    "mean_pinball_loss": "Mean pinball loss",
    "mae_p50": "MAE p50",
    "rmse_p50": "RMSE p50",
    "bias_p50": "Bias p50",
}

QCOL_RE = re.compile(r"^p(\d{1,2})$")
KEY_COLS = ["target_time_utc", "lead_time_h"]
ROW_INTERSECTION_KEY = "split,target,target_time_utc,lead_time_h"
DEFAULT_TRAINING_DAYS = 365.0
DEFAULT_EVAL_ORIGIN_START_UTC = "2025-01-13T23:00:00Z"
DEFAULT_EVAL_ORIGIN_END_UTC = "2026-02-26T21:00:00Z"
DA_PRICE_TARGETS = {"pred_da_price", "target_da_price"}
P50_TOLERANCE_THRESHOLDS = (1.0, 5.0, 10.0)
P50_TOLERANCE_UNIT_DA = "EUR/MWh"
PRICE_P50_TOLERANCE_CONFIG = {
    "pred_da_price": {
        "slug": "da_price",
        "label": "DA price",
        "thresholds": (1.0, 5.0, 10.0),
        "unit": "EUR/MWh",
        "unit_latex": "€/MWh",
    },
    "pred_afrr_capacity_price_pos": {
        "slug": "afrr_capacity_price_pos",
        "label": "aFRR capacity price +",
        "thresholds": (1.0, 5.0, 10.0),
        "unit": "EUR/MW",
        "unit_latex": "€/MW",
    },
    "pred_afrr_capacity_price_neg": {
        "slug": "afrr_capacity_price_neg",
        "label": "aFRR capacity price -",
        "thresholds": (1.0, 5.0, 10.0),
        "unit": "EUR/MW",
        "unit_latex": "€/MW",
    },
    "pred_afrr_activation_price_pos": {
        "slug": "afrr_activation_price_pos",
        "label": "aFRR activation price +",
        "thresholds": (10.0, 50.0, 100.0),
        "unit": "EUR/MWh",
        "unit_latex": "€/MWh",
    },
    "pred_afrr_activation_price_neg": {
        "slug": "afrr_activation_price_neg",
        "label": "aFRR activation price -",
        "thresholds": (10.0, 50.0, 100.0),
        "unit": "EUR/MWh",
        "unit_latex": "€/MWh",
    },
}
PRICE_P50_TOLERANCE_TARGETS = tuple(PRICE_P50_TOLERANCE_CONFIG)


def _p50_tolerance_title_label(target: str) -> str:
    labels = {
        "pred_da_price": "DA Price",
        "pred_afrr_capacity_price_pos": "aFRR Capacity Price Positive",
        "pred_afrr_capacity_price_neg": "aFRR Capacity Price Negative",
        "pred_afrr_activation_price_pos": "aFRR Activation Price Positive",
        "pred_afrr_activation_price_neg": "aFRR Activation Price Negative",
    }
    return labels.get(str(target), thesis_titlecase(_target_label(target)))


def _p50_tolerance_title(target: str) -> str:
    return f"Cumulative Absolute P50 Error Tolerance for {_p50_tolerance_title_label(target)}"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str


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
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(ModelSpec(canonical, label))
    if not out:
        raise ValueError("At least one model is required.")
    return out


def _discover_benchmark_dirs(benchmark_root: Path, explicit_dirs: list[Path]) -> list[Path]:
    if explicit_dirs:
        dirs = [p.resolve() for p in explicit_dirs]
    elif (benchmark_root / "diagnostics" / "joined_predictions").exists():
        dirs = [benchmark_root.resolve()]
    else:
        dirs = sorted(
            [p.resolve() for p in benchmark_root.iterdir() if (p / "diagnostics" / "joined_predictions").exists()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not dirs:
        raise FileNotFoundError(
            f"No benchmark directory with diagnostics/joined_predictions found under {benchmark_root}. "
            "Run scripts/run_forecast_benchmark.py with --save-joined-predictions first."
        )
    if len(dirs) > 1:
        names = ", ".join(str(p) for p in dirs[:5])
        raise ValueError(
            "Multiple benchmark directories were found. Pass --benchmark-dir explicitly to avoid mixing runs. "
            f"Candidates: {names}"
        )
    joined = dirs[0] / "diagnostics" / "joined_predictions"
    if not joined.exists():
        raise FileNotFoundError(f"Missing joined predictions directory: {joined}")
    return dirs


def _parse_joined_name(path: Path) -> tuple[str, str, str] | None:
    parts = path.stem.split("__", 2)
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def _quantile_cols(df: pd.DataFrame) -> dict[float, str]:
    out: dict[float, str] = {}
    for col in df.columns:
        m = QCOL_RE.match(str(col).lower())
        if not m:
            continue
        q = int(m.group(1)) / 100.0
        if 0.0 < q < 1.0:
            out[q] = str(col)
    return out


def _target_info(target: str) -> tuple[str, str]:
    return TARGET_GROUPS.get(target, ("Other", target.replace("_", " ")))


def _target_label(target: Any) -> str:
    return _target_info(str(target))[1]


def _read_joined_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    missing = [c for c in [*KEY_COLS, "y_true"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df["target_time_utc"] = pd.to_datetime(df["target_time_utc"], utc=True, errors="coerce")
    df["lead_time_h"] = pd.to_numeric(df["lead_time_h"], errors="coerce")
    if "p50" not in df.columns and "predicted_value" not in df.columns:
        raise ValueError(f"{path} must contain p50 or predicted_value for point diagnostics.")
    if "p50" not in df.columns:
        df["p50"] = pd.to_numeric(df["predicted_value"], errors="coerce")
    return df


def _parse_utc_bound(raw: str | None) -> pd.Timestamp | None:
    if raw is None or str(raw).strip() == "":
        return None
    return pd.to_datetime(raw, utc=True)


def _forecast_origin_utc(df: pd.DataFrame) -> pd.Series:
    if "forecast_time_utc" in df.columns:
        return pd.to_datetime(df["forecast_time_utc"], utc=True, errors="coerce")
    if "snapshot_time_utc" in df.columns:
        return pd.to_datetime(df["snapshot_time_utc"], utc=True, errors="coerce")
    return pd.to_datetime(df["target_time_utc"], utc=True, errors="coerce") - pd.to_timedelta(
        pd.to_numeric(df["lead_time_h"], errors="coerce"),
        unit="h",
    )


def _apply_forecast_origin_window(df: pd.DataFrame, *, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    if start is None and end is None:
        return df
    origin = _forecast_origin_utc(df)
    mask = origin.notna()
    if start is not None:
        mask &= origin.ge(start)
    if end is not None:
        mask &= origin.le(end)
    return df.loc[mask].copy()


def _row_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_time_utc": pd.to_datetime(df["target_time_utc"], utc=True, errors="coerce").astype("datetime64[ns, UTC]"),
            "lead_time_h": pd.to_numeric(df["lead_time_h"], errors="coerce").astype(float),
        },
        index=df.index,
    )


def _key_tuples(df: pd.DataFrame) -> set[tuple[pd.Timestamp, float]]:
    keys = _row_key_frame(df)
    return set(zip(keys["target_time_utc"], keys["lead_time_h"]))


def _assert_unique_keys(df: pd.DataFrame, *, model: str, split: str, target: str) -> None:
    duplicated = int(_row_key_frame(df).duplicated().sum())
    if duplicated:
        raise ValueError(
            f"Duplicate target_time_utc/lead_time_h rows for model={model}, split={split}, "
            f"target={target}: duplicated_rows={duplicated}"
        )


def _valid_frame(df: pd.DataFrame, qcols: dict[float, str]) -> pd.DataFrame:
    cols = list(dict.fromkeys([*KEY_COLS, "y_true", "p50", *[qcols[q] for q in sorted(qcols)]]))
    out = df[cols].copy()
    numeric_cols = list(dict.fromkeys(["y_true", "p50", *[qcols[q] for q in sorted(qcols)]]))
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    mask = pd.notna(out["target_time_utc"]) & pd.notna(out["lead_time_h"])
    for col in numeric_cols:
        mask &= np.isfinite(out[col].to_numpy(dtype=float))
    return out.loc[mask].copy()


def _pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    err = y - pred
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))


def _compute_metrics(df: pd.DataFrame, qcols: dict[float, str]) -> dict[str, float]:
    y = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    p50 = pd.to_numeric(df["p50"], errors="coerce").to_numpy(dtype=float)
    pinballs = [
        _pinball(y, pd.to_numeric(df[qcols[q]], errors="coerce").to_numpy(dtype=float), q)
        for q in sorted(qcols)
    ]
    return {
        "mean_pinball_loss": float(np.mean(pinballs)) if pinballs else float("nan"),
        "mae_p50": float(np.mean(np.abs(p50 - y))),
        "rmse_p50": float(np.sqrt(np.mean((p50 - y) ** 2))),
        "bias_p50": float(np.mean(p50 - y)),
    }


def build_full_metrics(
    *,
    benchmark_dir: Path,
    models: list[ModelSpec],
    splits: list[str],
    eval_origin_start: pd.Timestamp | None = None,
    eval_origin_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined_dir = benchmark_dir / "diagnostics" / "joined_predictions"
    files: dict[tuple[str, str, str], Path] = {}
    for path in sorted(joined_dir.glob("*.parquet")):
        parsed = _parse_joined_name(path)
        if parsed is not None:
            files[parsed] = path

    targets = sorted({target for _, split, target in files if split in splits}, key=target_sort_key)
    if not targets:
        raise FileNotFoundError(f"No joined prediction parquet files for splits={splits} in {joined_dir}.")

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for split in splits:
        for target in targets:
            loaded: dict[str, pd.DataFrame] = {}
            qmaps: dict[str, dict[float, str]] = {}
            for model in models:
                path = files.get((model.key, split, target))
                if path is None:
                    raise FileNotFoundError(
                        f"Missing joined predictions for model={model.key}, split={split}, target={target}. "
                        f"Expected file like {joined_dir / f'{model.key}__{split}__{target}.parquet'}."
                    )
                df = _apply_forecast_origin_window(
                    _read_joined_prediction(path),
                    start=eval_origin_start,
                    end=eval_origin_end,
                )
                _assert_unique_keys(df, model=model.key, split=split, target=target)
                loaded[model.key] = df
                qmaps[model.key] = _quantile_cols(df)

            common_qs = set.intersection(*(set(qmaps[m.key]) for m in models))
            if not common_qs:
                raise ValueError(f"No common quantile columns for split={split}, target={target}.")
            common_qcols = {m.key: {q: qmaps[m.key][q] for q in sorted(common_qs)} for m in models}

            valid_by_model = {m.key: _valid_frame(loaded[m.key], common_qcols[m.key]) for m in models}
            common_keys = set.intersection(*(_key_tuples(valid_by_model[m.key]) for m in models))
            if not common_keys:
                raise ValueError(f"Common valid row intersection is empty for split={split}, target={target}.")

            common_key_df = pd.DataFrame(list(common_keys), columns=KEY_COLS)
            common_key_df["target_time_utc"] = pd.to_datetime(common_key_df["target_time_utc"], utc=True)
            common_key_df["lead_time_h"] = pd.to_numeric(common_key_df["lead_time_h"], errors="coerce").astype(float)
            lead_values = pd.to_numeric(common_key_df["lead_time_h"], errors="coerce")
            target_group, target_label = _target_info(target)
            quantiles_used = ",".join(f"p{int(round(q * 100)):02d}" for q in sorted(common_qs))

            for model in models:
                own_valid = valid_by_model[model.key]
                eval_df = own_valid.merge(common_key_df, on=KEY_COLS, how="inner").sort_values(KEY_COLS)
                values = _compute_metrics(eval_df, common_qcols[model.key])
                n_obs = int(len(loaded[model.key]))
                n_valid_before = int(len(own_valid))
                n_common = int(len(eval_df))
                n_missing = int(n_obs - n_valid_before)
                n_dropped = int(n_valid_before - n_common)
                base = {
                    "model": model.key,
                    "model_label": model.label,
                    "split": split,
                    "target": target,
                    "target_label": target_label,
                    "target_group": target_group,
                    "n_obs": n_obs,
                    "n_valid_rows": n_common,
                    "n_missing_rows": n_missing,
                    "n_dropped_rows": n_dropped,
                    "lead_min": float(lead_values.min()),
                    "lead_max": float(lead_values.max()),
                    "quantiles_used": quantiles_used,
                    "row_intersection_key": ROW_INTERSECTION_KEY,
                    "eval_origin_start_utc": eval_origin_start.isoformat() if eval_origin_start is not None else "",
                    "eval_origin_end_utc": eval_origin_end.isoformat() if eval_origin_end is not None else "",
                }
                for metric in METRICS:
                    rows.append(
                        {
                            **base,
                            "metric": metric,
                            "value": values[metric],
                            "lower_is_better": bool(LOWER_IS_BETTER[metric]),
                        }
                    )
                diagnostics.append(
                    {
                        "split": split,
                        "target": target,
                        "model": model.key,
                        "original_valid_rows": n_valid_before,
                        "common_intersection_rows": n_common,
                        "dropped_rows": n_dropped,
                        "quantiles_available": ",".join(f"p{int(round(q * 100)):02d}" for q in sorted(qmaps[model.key])),
                        "quantiles_used": quantiles_used,
                        "lead_min": float(lead_values.min()),
                        "lead_max": float(lead_values.max()),
                        "eval_origin_start_utc": eval_origin_start.isoformat() if eval_origin_start is not None else "",
                        "eval_origin_end_utc": eval_origin_end.isoformat() if eval_origin_end is not None else "",
                    }
                )

    metrics = sort_target_frame(pd.DataFrame(rows), target_col="target", extra_cols=["split", "metric", "model_label"])
    diag = sort_target_frame(pd.DataFrame(diagnostics), target_col="target", extra_cols=["split", "model"])
    return metrics, diag


def _threshold_slug(threshold: float) -> str:
    x = float(threshold)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return str(x).replace(".", "_")


def _valid_p50_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[*KEY_COLS, "y_true", "p50"]].copy()
    for col in ["y_true", "p50"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    mask = pd.notna(out["target_time_utc"]) & pd.notna(out["lead_time_h"])
    mask &= np.isfinite(out["y_true"].to_numpy(dtype=float))
    mask &= np.isfinite(out["p50"].to_numpy(dtype=float))
    return out.loc[mask].copy()


def build_p50_error_tolerance_outputs(
    *,
    benchmark_dir: Path,
    models: list[ModelSpec],
    splits: list[str],
    target: str = "pred_da_price",
    eval_origin_start: pd.Timestamp | None = None,
    eval_origin_end: pd.Timestamp | None = None,
    thresholds: tuple[float, ...] = P50_TOLERANCE_THRESHOLDS,
    x_max: float | None = None,
    grid_size: int = 201,
    unit: str = P50_TOLERANCE_UNIT_DA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined_dir = benchmark_dir / "diagnostics" / "joined_predictions"
    files: dict[tuple[str, str, str], Path] = {}
    for path in sorted(joined_dir.glob("*.parquet")):
        parsed = _parse_joined_name(path)
        if parsed is not None:
            files[parsed] = path

    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for split in splits:
        loaded: dict[str, pd.DataFrame] = {}
        for model in models:
            path = files.get((model.key, split, target))
            if path is None:
                raise FileNotFoundError(
                    f"Missing joined predictions for p50 tolerance curve: model={model.key}, split={split}, target={target}. "
                    f"Expected {joined_dir / f'{model.key}__{split}__{target}.parquet'}."
                )
            df = _apply_forecast_origin_window(
                _read_joined_prediction(path),
                start=eval_origin_start,
                end=eval_origin_end,
            )
            _assert_unique_keys(df, model=model.key, split=split, target=target)
            loaded[model.key] = _valid_p50_frame(df)

        common_keys = set.intersection(*(_key_tuples(loaded[m.key]) for m in models))
        if not common_keys:
            raise ValueError(f"Common p50 valid row intersection is empty for split={split}, target={target}.")
        common_key_df = pd.DataFrame(list(common_keys), columns=KEY_COLS)
        common_key_df["target_time_utc"] = pd.to_datetime(common_key_df["target_time_utc"], utc=True)
        common_key_df["lead_time_h"] = pd.to_numeric(common_key_df["lead_time_h"], errors="coerce").astype(float)

        abs_errors_by_model: dict[str, np.ndarray] = {}
        pooled_abs_errors: list[np.ndarray] = []
        for model in models:
            eval_df = loaded[model.key].merge(common_key_df, on=KEY_COLS, how="inner").sort_values(KEY_COLS)
            err = (
                pd.to_numeric(eval_df["p50"], errors="coerce").to_numpy(dtype=float)
                - pd.to_numeric(eval_df["y_true"], errors="coerce").to_numpy(dtype=float)
            )
            abs_err = np.abs(err)
            abs_errors_by_model[model.label] = abs_err
            pooled_abs_errors.append(abs_err)

        pooled = np.concatenate(pooled_abs_errors) if pooled_abs_errors else np.array([], dtype=float)
        finite_pooled = pooled[np.isfinite(pooled)]
        if finite_pooled.size == 0:
            raise ValueError(f"No finite absolute p50 errors for split={split}, target={target}.")
        robust_max = float(np.nanpercentile(finite_pooled, 95.0)) if x_max is None else float(x_max)
        robust_max = max(robust_max, max(thresholds), 1.0)
        grid = np.linspace(0.0, robust_max, int(max(grid_size, 2)))
        threshold_values = sorted(set(float(x) for x in [*grid.tolist(), *thresholds, 0.0] if 0.0 <= float(x) <= robust_max))

        for model in models:
            abs_err = abs_errors_by_model[model.label]
            finite = abs_err[np.isfinite(abs_err)]
            n_obs = int(finite.size)
            if n_obs == 0:
                continue
            for threshold in threshold_values:
                curve_rows.append(
                    {
                        "split": split,
                        "target": target,
                        "model": model.label,
                        "threshold": float(threshold),
                        "share_within_threshold": float(np.mean(finite <= threshold)),
                        "n_obs": n_obs,
                        "unit": unit,
                    }
                )
            summary = {
                "split": split,
                "target": target,
                "target_label": _target_label(target),
                "model": model.label,
                "median_absolute_error": float(np.median(finite)),
                "mae_p50": float(np.mean(finite)),
                "absolute_error_p80": float(np.nanpercentile(finite, 80.0)),
                "n_obs": n_obs,
                "unit": unit,
            }
            for threshold in thresholds:
                summary[f"share_le_{_threshold_slug(threshold)}"] = float(np.mean(finite <= float(threshold)))
            summary_rows.append(summary)

    curve = pd.DataFrame(curve_rows)
    summary = pd.DataFrame(summary_rows)
    order = {label: i for i, label in enumerate(MODEL_LABELS)}
    if not curve.empty:
        curve["_model_order"] = curve["model"].map(order)
        curve = curve.sort_values(["split", "target", "_model_order", "threshold"]).drop(columns=["_model_order"]).reset_index(drop=True)
    if not summary.empty:
        summary["_model_order"] = summary["model"].map(order)
        summary = summary.sort_values(["split", "target", "_model_order"]).drop(columns=["_model_order"]).reset_index(drop=True)
    return curve, summary


def _collect_common_p50_abs_errors(
    *,
    benchmark_dir: Path,
    models: list[ModelSpec],
    split: str,
    target: str,
    eval_origin_start: pd.Timestamp | None = None,
    eval_origin_end: pd.Timestamp | None = None,
) -> list[np.ndarray]:
    joined_dir = benchmark_dir / "diagnostics" / "joined_predictions"
    files: dict[tuple[str, str, str], Path] = {}
    for path in sorted(joined_dir.glob("*.parquet")):
        parsed = _parse_joined_name(path)
        if parsed is not None:
            files[parsed] = path
    loaded: dict[str, pd.DataFrame] = {}
    for model in models:
        path = files.get((model.key, split, target))
        if path is None:
            raise FileNotFoundError(f"Missing joined predictions for pooled p80: model={model.key}, split={split}, target={target}.")
        df = _apply_forecast_origin_window(
            _read_joined_prediction(path),
            start=eval_origin_start,
            end=eval_origin_end,
        )
        _assert_unique_keys(df, model=model.key, split=split, target=target)
        loaded[model.key] = _valid_p50_frame(df)
    common_keys = set.intersection(*(_key_tuples(loaded[m.key]) for m in models))
    if not common_keys:
        return []
    common_key_df = pd.DataFrame(list(common_keys), columns=KEY_COLS)
    common_key_df["target_time_utc"] = pd.to_datetime(common_key_df["target_time_utc"], utc=True)
    common_key_df["lead_time_h"] = pd.to_numeric(common_key_df["lead_time_h"], errors="coerce").astype(float)
    arrays: list[np.ndarray] = []
    for model in models:
        eval_df = loaded[model.key].merge(common_key_df, on=KEY_COLS, how="inner").sort_values(KEY_COLS)
        err = (
            pd.to_numeric(eval_df["p50"], errors="coerce").to_numpy(dtype=float)
            - pd.to_numeric(eval_df["y_true"], errors="coerce").to_numpy(dtype=float)
        )
        finite = np.abs(err)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            arrays.append(finite)
    return arrays


def build_price_p50_error_tolerance_outputs(
    *,
    benchmark_dir: Path,
    models: list[ModelSpec],
    splits: list[str],
    eval_origin_start: pd.Timestamp | None = None,
    eval_origin_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curves: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    global_abs_errors: dict[str, list[np.ndarray]] = {split: [] for split in splits}
    for target, cfg in PRICE_P50_TOLERANCE_CONFIG.items():
        curve, summary = build_p50_error_tolerance_outputs(
            benchmark_dir=benchmark_dir,
            models=models,
            splits=splits,
            target=target,
            eval_origin_start=eval_origin_start,
            eval_origin_end=eval_origin_end,
            thresholds=tuple(float(x) for x in cfg["thresholds"]),
            unit=str(cfg["unit"]),
        )
        curves.append(curve)
        summaries.append(summary)
        for split in splits:
            global_abs_errors[split].extend(
                _collect_common_p50_abs_errors(
                    benchmark_dir=benchmark_dir,
                    models=models,
                    split=split,
                    target=target,
                    eval_origin_start=eval_origin_start,
                    eval_origin_end=eval_origin_end,
                )
            )
    curve_all = pd.concat(curves, ignore_index=True)
    summary_all = pd.concat(summaries, ignore_index=True)
    global_p80 = {
        split: float(np.nanpercentile(np.concatenate(arrays), 80.0))
        for split, arrays in global_abs_errors.items()
        if arrays
    }
    performance = build_p50_error_tolerance_performance_values(summary_all, splits=splits, global_p80=global_p80)
    return curve_all, summary_all, performance


def build_p50_error_tolerance_performance_values(summary: pd.DataFrame, *, splits: list[str], global_p80: dict[str, float] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if summary.empty:
        return pd.DataFrame()
    for _, row in summary.iterrows():
        target = str(row["target"])
        cfg = PRICE_P50_TOLERANCE_CONFIG.get(target)
        if cfg is None:
            continue
        out = {
            "split": row["split"],
            "target": target,
            "target_label": cfg["label"],
            "model": row["model"],
            "unit": cfg["unit"],
            "n_obs": int(row["n_obs"]),
            "median_absolute_error": float(row["median_absolute_error"]),
            "mae_p50": float(row["mae_p50"]),
            "absolute_error_p80": float(row["absolute_error_p80"]),
        }
        for threshold in cfg["thresholds"]:
            slug = _threshold_slug(float(threshold))
            out[f"share_le_{slug}"] = float(row.get(f"share_le_{slug}", np.nan))
            out[f"share_le_{slug}_pct"] = 100.0 * float(row.get(f"share_le_{slug}", np.nan))
        rows.append(out)
    perf = pd.DataFrame(rows)
    if perf.empty:
        return perf
    global_rows: list[dict[str, Any]] = []
    for split in splits:
        d = perf[perf["split"].eq(split)].copy()
        d["n_obs"] = pd.to_numeric(d["n_obs"], errors="coerce")
        d["absolute_error_p80"] = pd.to_numeric(d["absolute_error_p80"], errors="coerce")
        d = d.dropna(subset=["n_obs", "absolute_error_p80"])
        if d.empty:
            continue
        global_rows.append(
            {
                "split": split,
                "target": "ALL_PRICE_TARGETS_POOLED",
                "target_label": "All price targets",
                "model": "ALL_MODELS",
                "unit": "mixed price units",
                "n_obs": int(d["n_obs"].sum()),
                "median_absolute_error": np.nan,
                "mae_p50": np.nan,
                "absolute_error_p80": (global_p80 or {}).get(split, np.nan),
                "note": "Exact pooled p80 over all price-target common-row absolute p50 errors and all models; target units differ between EUR/MWh and EUR/MW.",
            }
        )
    if global_rows:
        perf = pd.concat([perf, pd.DataFrame(global_rows)], ignore_index=True)
    return perf


def _latex_escape(value: Any) -> str:
    s = str(value)
    minus_token = "@@RQ1MINUS@@"
    for label in ["aFRR capacity price", "aFRR activation price", "aFRR activation rate"]:
        s = s.replace(f"{label} positive", f"{label} +")
        s = s.replace(f"{label} Positive", f"{label} +")
        s = s.replace(f"{label} negative", f"{label} {minus_token}")
        s = s.replace(f"{label} Negative", f"{label} {minus_token}")
        s = s.replace(f"{label} -", f"{label} {minus_token}")
        s = s.replace(f"{label} \u2212", f"{label} {minus_token}")
    repl = {
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
    }
    for old, new in repl.items():
        s = s.replace(old, new)
    return s.replace(minus_token, "$-$")


def _fmt_value(value: Any, *, pct: bool = False, decimals: int = 4) -> str:
    try:
        x = float(value)
    except Exception:
        return "-"
    if not np.isfinite(x):
        return "-"
    return f"{x:.2f}" if pct else f"{x:.{decimals}f}"


def _fmt_blankable(value: Any, *, pct: bool = False) -> str:
    try:
        x = float(value)
    except Exception:
        return "--"
    if not np.isfinite(x):
        return "--"
    return f"{x:.2f}" if pct else f"{x:.4f}"


def _fmt_int(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "-"
    if not np.isfinite(x):
        return "-"
    return f"{int(round(x)):,}"


def _caption_n_suffix(table: pd.DataFrame) -> str:
    if "n_obs" not in table.columns:
        return ""
    vals = sorted({int(v) for v in table["n_obs"].dropna().to_list() if np.isfinite(float(v))})
    if not vals:
        return ""
    if len(vals) == 1:
        return f" (N = {_fmt_int(vals[0])})"
    return f" (N = {_fmt_int(vals[0])}-{_fmt_int(vals[-1])})"


def _latex_header(label: str) -> str:
    raw = str(label)
    body = raw if raw.startswith("\\") else _latex_escape(raw)
    return r"\textbf{" + body + "}"


def _pivot_metric(metrics: pd.DataFrame, *, split: str, metric: str) -> pd.DataFrame:
    d = metrics.loc[(metrics["split"] == split) & (metrics["metric"] == metric)].copy()
    if d.empty:
        return pd.DataFrame()
    idx_cols = ["target_group", "target", "metric", "n_valid_rows", "quantiles_used", "lead_min", "lead_max"]
    pivot = (
        d.pivot_table(index=idx_cols, columns="model_label", values="value", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in MODEL_LABELS:
        if col not in pivot.columns:
            pivot[col] = np.nan
    return pivot


def _add_best_and_improvement(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    best_models: list[str] = []
    rel_improvements: list[float] = []
    for _, row in out.iterrows():
        metric = str(row.get("metric", ""))
        if not LOWER_IS_BETTER.get(metric, False):
            best_models.append("")
            rel_improvements.append(float("nan"))
            continue
        vals = {
            label: float(row[label])
            for label in MODEL_LABELS
            if label in row and pd.notna(row[label]) and np.isfinite(float(row[label]))
        }
        if not vals:
            best_models.append("")
            rel_improvements.append(float("nan"))
            continue
        best = min(vals, key=vals.get)
        rlqr = vals.get("RLQR", float("nan"))
        best_val = vals[best]
        rel = 100.0 * (rlqr - best_val) / abs(rlqr) if np.isfinite(rlqr) and abs(rlqr) > 1e-12 else float("nan")
        best_models.append(best)
        rel_improvements.append(float(rel))
    out["best_model"] = best_models
    out["relative_improvement_vs_RLQR_pct"] = rel_improvements
    return out


def build_primary_table(metrics: pd.DataFrame, *, split: str) -> pd.DataFrame:
    primary = _add_best_and_improvement(_pivot_metric(metrics, split=split, metric="mean_pinball_loss"))
    if primary.empty:
        return primary
    primary = primary.rename(columns={"n_valid_rows": "n_obs"})
    cols = [
        "target",
        "RLQR",
        "XGB",
        "TFT",
        "best_model",
        "relative_improvement_vs_RLQR_pct",
        "n_obs",
        "quantiles_used",
        "lead_min",
        "lead_max",
    ]
    return sort_target_frame(primary[cols], target_col="target")


def build_detailed_table(metrics: pd.DataFrame, *, split: str) -> pd.DataFrame:
    parts = [_pivot_metric(metrics, split=split, metric=m) for m in METRICS]
    d = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
    if d.empty:
        return d
    d = _add_best_and_improvement(d).rename(columns={"n_valid_rows": "n_obs"})
    cols = [
        "target",
        "metric",
        "RLQR",
        "XGB",
        "TFT",
        "best_model",
        "relative_improvement_vs_RLQR_pct",
        "n_obs",
        "quantiles_used",
        "lead_min",
        "lead_max",
    ]
    metric_order = {m: i for i, m in enumerate(METRICS)}
    d["_metric_order"] = d["metric"].map(metric_order)
    return sort_target_frame(d[cols + ["_metric_order"]], target_col="target", extra_cols=["_metric_order"]).drop(columns=["_metric_order"]).reset_index(drop=True)


def _latex_table(
    table: pd.DataFrame,
    *,
    columns: list[str],
    headers: list[str],
    caption: str,
    label: str,
    path: Path,
    bold_best_model_values: bool = False,
    value_decimals: int = 4,
    activation_rate_value_decimals: int | None = None,
) -> None:
    d = table[columns].copy()
    if len(headers) != len(columns):
        raise ValueError("LaTeX headers and columns must have the same length.")
    colspec = "@{}" + "l" + ("r" * (len(columns) - 1)) + "@{}"
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        rf"    \begin{{tabular}}{{{colspec}}}",
        r"        \toprule",
        "        " + " & ".join(_latex_header(c) for c in headers) + r" \\",
        r"        \midrule",
    ]
    for _, row in d.iterrows():
        vals: list[str] = []
        for col in d.columns:
            val = row[col]
            if col in {"TFT", "XGB", "RLQR"}:
                decimals = value_decimals
                if activation_rate_value_decimals is not None and "activation_rate" in str(row.get("target", "")):
                    decimals = activation_rate_value_decimals
                formatted = _fmt_value(val, decimals=decimals)
                if bold_best_model_values and str(row.get("best_model", "")) == col and formatted != "-":
                    formatted = rf"\textbf{{{formatted}}}"
                vals.append(formatted)
            elif col in {
                "final_training_observed_hours",
                "final_training_scaled_hours_365d",
                "hpo_observed_hours_display",
                "hpo_scaled_hours_display",
                "training_days_h1",
            }:
                formatted = _fmt_value(val)
                vals.append(_latex_escape(val) if formatted == "-" and str(val).strip() else formatted)
            elif col == "relative_improvement_vs_RLQR_pct":
                vals.append(_fmt_blankable(val, pct=True))
            elif col == "n_obs":
                vals.append(str(int(val)) if pd.notna(val) else "-")
            elif col == "target":
                vals.append(_latex_escape(_target_label(val)))
            elif col == "metric":
                vals.append(_latex_escape(METRIC_LABELS.get(str(val), str(val))))
            elif col == "best_model" and (pd.isna(val) or str(val) == ""):
                vals.append("--")
            else:
                vals.append(_latex_escape(val))
        lines.append("        " + " & ".join(vals) + r" \\")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            rf"    \caption{{{_latex_escape(caption)}}}",
            rf"    \label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_tables(primary: pd.DataFrame, detailed: pd.DataFrame, *, out_dir: Path, split: str) -> list[Path]:
    latex_dir = out_dir / "latex"
    primary_path = latex_dir / f"rq1_4_1_1_forecast_metrics_full_primary_{split}.tex"
    detailed_path = latex_dir / f"rq1_4_1_1_forecast_metrics_full_detailed_{split}.tex"
    _latex_table(
        primary,
        columns=["target", "RLQR", "XGB", "TFT", "best_model", "relative_improvement_vs_RLQR_pct"],
        headers=["Target", "RLQR", "XGB", "TFT", "Best model", r"\shortstack{Improvement\\vs RLQR (\%)}"],
        caption="Model Mean Pinball Loss",
        label="tab:forecast_metrics_full_primary",
        path=primary_path,
        bold_best_model_values=True,
        value_decimals=2,
        activation_rate_value_decimals=5,
    )
    _latex_table(
        detailed,
        columns=["target", "metric", "RLQR", "XGB", "TFT", "best_model", "relative_improvement_vs_RLQR_pct", "n_obs"],
        headers=["Target", "Metric", "RLQR", "XGB", "TFT", "Best model", r"\shortstack{Improvement\\vs RLQR (\%)}", "N"],
        caption=(
            "Detailed full unweighted forecast metrics on the test split. "
            "Lower values are better for mean pinball loss, MAE and RMSE. "
            "Bias p50 is the mean p50 forecast error and indicates systematic over- or underprediction."
        ),
        label="tab:forecast_metrics_full_detailed",
        path=detailed_path,
    )
    return [primary_path, detailed_path]


def _resolve_manifest_pointer(path: Path) -> Path | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return path.resolve()
    manifest_path = data.get("manifest_path_abs") or data.get("manifest_path")
    if manifest_path:
        p = Path(str(manifest_path))
        if not p.is_absolute():
            p = (path.parent / p).resolve()
        if p.exists():
            return p.resolve()
    if "training" in data:
        return path.resolve()
    return None


def _resolve_model_manifest_paths(benchmark_dir: Path) -> list[Path]:
    input_manifest = benchmark_dir / "input_manifest.json"
    raw_paths: list[Any] = []
    try:
        data = json.loads(input_manifest.read_text(encoding="utf-8")) if input_manifest.exists() else {}
        raw_paths.extend(data.get("model_run_manifests", []))
    except Exception:
        pass
    if not raw_paths:
        raw_paths.extend(
            [
                ROOT / "artifacts/model_runs/latest_xgboost.json",
                ROOT / "artifacts/model_runs/latest_linear.json",
                ROOT / "artifacts/model_runs/latest_tft.json",
            ]
        )
    paths: list[Path] = []
    for raw in raw_paths:
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            p = (benchmark_dir / p).resolve()
            if not p.exists():
                p = (ROOT / str(raw)).resolve()
        resolved = _resolve_manifest_pointer(p)
        if resolved and resolved.exists():
            paths.append(resolved.resolve())
    deduped: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    return deduped


def _model_label_from_type(model_type: str, run_id: str = "") -> tuple[str, str]:
    key = str(model_type or run_id).strip().lower()
    if "xgb" in key or "xgboost" in key:
        return "xgb", "XGB"
    if "tft" in key:
        return "tft", "TFT"
    if "linear" in key or "rlqr" in key:
        return "linear", "RLQR"
    return key or "unknown", str(model_type or run_id or "unknown")


def _hpo_duration_for_model(manifest: dict[str, Any], manifest_path: Path, *, train_rows_h1: float | None) -> dict[str, Any]:
    hpo_map = manifest.get("training", {}).get("hpo_artifacts_by_target", {})
    total_s = 0.0
    trials_with_timing = 0
    trial_rows_without_timing = 0
    trial_files = 0
    for raw in hpo_map.values() if isinstance(hpo_map, dict) else []:
        hpo_json = Path(str(raw))
        if not hpo_json.is_absolute():
            hpo_json = (manifest_path.parent / hpo_json).resolve()
        trials_csv = hpo_json.with_name(hpo_json.stem + "_trials.csv")
        if not trials_csv.exists():
            continue
        trial_files += 1
        try:
            df = pd.read_csv(trials_csv)
        except Exception:
            continue
        if "duration_seconds" not in df.columns:
            trial_rows_without_timing += int(len(df))
            continue
        vals = pd.to_numeric(df["duration_seconds"], errors="coerce").dropna()
        total_s += float(vals.sum())
        trials_with_timing += int(vals.shape[0])
        trial_rows_without_timing += int(len(df) - vals.shape[0])
    factor = (DEFAULT_TRAINING_DAYS * 24.0 / train_rows_h1) if train_rows_h1 and train_rows_h1 > 0 else np.nan
    if trials_with_timing:
        status = "directly recorded"
    elif trial_files:
        status = "trials recorded without timing"
    else:
        status = "not recorded"
    return {
        "hpo_observed_hours": total_s / 3600.0 if trials_with_timing else np.nan,
        "hpo_scaled_hours_365d": total_s * factor / 3600.0 if trials_with_timing and np.isfinite(factor) else np.nan,
        "hpo_trials_with_timing": trials_with_timing,
        "hpo_trial_files_found": trial_files,
        "hpo_trial_rows_without_timing": trial_rows_without_timing,
        "hpo_timing_status": status,
    }


def build_computational_cost_table(benchmark_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for manifest_path in _resolve_model_manifest_paths(benchmark_dir):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_dir = manifest_path.parent
        ctx_path = run_dir / str(manifest.get("training", {}).get("context_path", "training_run_context.json"))
        if not ctx_path.exists():
            continue
        try:
            context = json.loads(ctx_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        model_key, model_label = _model_label_from_type(str(manifest.get("training", {}).get("model_type", "")), str(manifest.get("run_id", "")))
        data_report_path = run_dir / str(manifest.get("training", {}).get("data_integrity_report_path", "data_integrity_report.json"))
        train_rows = np.nan
        train_start = ""
        train_end = ""
        if data_report_path.exists():
            try:
                report = json.loads(data_report_path.read_text(encoding="utf-8"))
                train = report.get("da_base_dir_report", {}).get("bundles", {}).get("da", {}).get("files", {}).get("train", {})
                train_rows = float(train.get("rows", np.nan))
                train_start = str(train.get("timestamp_min", ""))
                train_end = str(train.get("timestamp_max", ""))
            except Exception:
                pass
        factor = (DEFAULT_TRAINING_DAYS * 24.0 / train_rows) if np.isfinite(train_rows) and train_rows > 0 else np.nan
        duration_s = float(context.get("duration_seconds", np.nan))
        row = {
            "model": model_key,
            "model_label": model_label,
            "run_id": manifest.get("run_id", run_dir.name),
            "final_training_observed_hours": duration_s / 3600.0 if np.isfinite(duration_s) else np.nan,
            "final_training_scaled_hours_365d": duration_s * factor / 3600.0 if np.isfinite(duration_s) and np.isfinite(factor) else np.nan,
            "training_rows_h1": train_rows,
            "training_days_h1": train_rows / 24.0 if np.isfinite(train_rows) else np.nan,
            "scaling_factor_to_365d": factor,
            "train_start_utc": train_start,
            "train_end_utc": train_end,
            "runtime_source": str(ctx_path),
        }
        row.update(_hpo_duration_for_model(manifest, manifest_path, train_rows_h1=train_rows if np.isfinite(train_rows) else None))
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(by="model", key=lambda s: s.map(lambda value: model_sort_key(value)[0])).reset_index(drop=True)


def write_computational_cost_latex_table(costs: pd.DataFrame, *, out_dir: Path) -> Path | None:
    if costs.empty:
        return None
    table = costs.copy()
    table = sort_model_frame(table, model_col="model_label")
    path = out_dir / "latex" / "rq1_4_1_1_computational_cost_365d.tex"
    _latex_table(
        table,
        columns=[
            "model_label",
            "final_training_observed_hours",
            "final_training_scaled_hours_365d",
        ],
        headers=[
            "Model",
            "Training observed (h)",
            "Training scaled to 365d (h)",
        ],
        caption="Computational Cost of Each Model Scaled for 365 Days of Training Data.",
        label="tab:rq1-4-1-1-computational-cost-365d",
        path=path,
    )
    return path


def write_computational_cost_figure(costs: pd.DataFrame, *, out_dir: Path, skip_pdf: bool = False) -> list[Path]:
    if costs.empty:
        return []
    import matplotlib.pyplot as plt

    d = costs.dropna(subset=["final_training_scaled_hours_365d"]).copy()
    if d.empty:
        return []
    d = sort_model_frame(d, model_col="model_label")
    apply_geo_style()
    colors = [get_model_color(str(m)) for m in d["model"]]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    training_bars = ax.bar(
        d["model_label"],
        d["final_training_scaled_hours_365d"],
        color=colors,
        edgecolor=colors,
        width=0.62,
        zorder=3,
        label="Training",
    )
    ax.set_ylabel("Time (h)")
    ax.set_title(thesis_titlecase("Computational Cost of Each Model Scaled for 365 Days of Training Data"))
    ax.bar_label(training_bars, labels=[f"{v:.2f}" for v in d["final_training_scaled_hours_365d"]], padding=4, fontsize=9)
    ax.set_ylim(0, max(0.1, float(d["final_training_scaled_hours_365d"].max()) * 1.18))
    ax.text(
        0.01,
        -0.18,
        "Scaling: observed final-training wall-clock time x (365 days / observed hourly train-days).",
        transform=ax.transAxes,
        fontsize=8.5,
        color=THESIS_PALETTE["neutral_dark"],
        va="top",
    )
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [fig_dir / "rq1_4_1_1_computational_cost_365d.png"]
    if not skip_pdf:
        paths.append(fig_dir / "rq1_4_1_1_computational_cost_365d.pdf")
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


def write_relative_pinball_figure(primary: pd.DataFrame, *, out_dir: Path, split: str, skip_pdf: bool = False) -> list[Path]:
    import matplotlib.pyplot as plt

    d = sort_target_frame(primary, target_col="target")
    for col in MODEL_LABELS:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d[np.isfinite(d["RLQR"]) & (d["RLQR"].abs() > 1e-12)].copy()
    if d.empty:
        return []
    apply_geo_style()
    labels = [_target_label(t) for t in d["target"]]
    x = np.arange(len(d), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(1.0, color=get_model_color("linear"), linewidth=1.3, linestyle=":", label="RLQR")
    for model, offset in [("XGB", -width / 2), ("TFT", width / 2)]:
        rel = d[model].to_numpy(dtype=float) / d["RLQR"].to_numpy(dtype=float)
        color = get_model_color(model.lower())
        ax.bar(x + offset, rel, width=width, label=model, color=color, edgecolor=color, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean pinball loss relative to RLQR")
    ax.set_title(thesis_titlecase("Mean pinball loss by target relative to RLQR"))
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / f"rq1_4_1_1_forecast_metrics_full_relative_pinball_{split}.png"
    fig.savefig(png)
    paths = [png]
    if not skip_pdf:
        pdf = fig_dir / f"rq1_4_1_1_forecast_metrics_full_relative_pinball_{split}.pdf"
        fig.savefig(pdf)
        paths.append(pdf)
    plt.close(fig)
    return paths


def write_relative_mae_figure(detailed: pd.DataFrame, *, out_dir: Path, split: str, skip_pdf: bool = False) -> list[Path]:
    import matplotlib.pyplot as plt

    d = sort_target_frame(detailed.loc[detailed["metric"].eq("mae_p50")].copy(), target_col="target")
    for col in MODEL_LABELS:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d[np.isfinite(d["RLQR"]) & (d["RLQR"].abs() > 1e-12)].copy()
    if d.empty:
        return []
    apply_geo_style()
    labels = [_target_label(t) for t in d["target"]]
    x = np.arange(len(d), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(1.0, color=get_model_color("linear"), linewidth=1.3, linestyle=":", label="RLQR")
    for model, offset in [("XGB", -width / 2), ("TFT", width / 2)]:
        rel = d[model].to_numpy(dtype=float) / d["RLQR"].to_numpy(dtype=float)
        color = get_model_color(model.lower())
        ax.bar(x + offset, rel, width=width, label=model, color=color, edgecolor=color, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("MAE p50 relative to RLQR")
    ax.set_title(thesis_titlecase("Relative MAE p50 by target (RLQR = 1; lower is better)"))
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / f"rq1_4_1_1_forecast_metrics_full_relative_mae_p50_{split}.png"
    fig.savefig(png)
    paths = [png]
    if not skip_pdf:
        pdf = fig_dir / f"rq1_4_1_1_forecast_metrics_full_relative_mae_p50_{split}.pdf"
        fig.savefig(pdf)
        paths.append(pdf)
    plt.close(fig)
    return paths


def write_p50_error_tolerance_figure(curve: pd.DataFrame, *, out_dir: Path, split: str, skip_pdf: bool = False) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    paths: list[Path] = []
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    apply_geo_style()
    for target, cfg in PRICE_P50_TOLERANCE_CONFIG.items():
        d = curve.loc[(curve["split"].eq(split)) & (curve["target"].eq(target))].copy()
        if d.empty:
            continue
        d["threshold"] = pd.to_numeric(d["threshold"], errors="coerce")
        d["share_within_threshold"] = pd.to_numeric(d["share_within_threshold"], errors="coerce")
        d = d.dropna(subset=["threshold", "share_within_threshold"])
        if d.empty:
            continue

        fig, ax = plt.subplots(figsize=(8.8, 5.4))
        for label in MODEL_LABELS:
            part = d[d["model"].eq(label)].sort_values("threshold")
            if part.empty:
                continue
            model_key = "linear" if label == "RLQR" else label.lower()
            ax.plot(
                part["threshold"].to_numpy(dtype=float),
                part["share_within_threshold"].to_numpy(dtype=float),
                label=label,
                color=get_model_color(model_key),
                linewidth=2.0,
            )
        x_max = float(d["threshold"].max())
        for ref in cfg["thresholds"]:
            ref_f = float(ref)
            if 0.0 <= ref_f <= x_max:
                ax.axvline(ref_f, color=THESIS_PALETTE["neutral_dark"], linestyle="--", linewidth=0.9, alpha=0.65)
                ax.text(ref_f, 0.03, f"{int(ref_f) if ref_f.is_integer() else ref_f:g} {cfg['unit_latex']}", rotation=90, va="bottom", ha="right", fontsize=8)
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_xlabel(f"Absolute p50 error threshold in {cfg['unit_latex']}")
        ax.set_ylabel("Share Within Threshold")
        ax.set_title(_p50_tolerance_title(target), pad=18)
        fig.text(
            0.125,
            0.91,
            "Higher is better. The curve shows the share of p50 forecasts within each absolute error threshold.",
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.legend(ncol=3, loc="lower right")
        ax.grid(alpha=0.25)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        stem = f"rq1_4_1_1_{cfg['slug']}_p50_absolute_error_tolerance_curve"
        png = fig_dir / f"{stem}.png"
        fig.savefig(png)
        paths.append(png)
        if not skip_pdf:
            pdf = fig_dir / f"{stem}.pdf"
            fig.savefig(pdf)
            paths.append(pdf)
        plt.close(fig)
    return paths


def write_p50_error_tolerance_latex_figures(*, out_dir: Path, split: str) -> list[Path]:
    latex_dir = out_dir / "latex_figures"
    latex_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for target, cfg in PRICE_P50_TOLERANCE_CONFIG.items():
        stem = f"{cfg['slug']}_p50_absolute_error_tolerance_curve"
        path = latex_dir / f"rq1_4_1_1_{stem}.tex"
        caption = _p50_tolerance_title(target)
        lines = [
            r"\begin{figure}[htbp]",
            r"    \centering",
            rf"    \includegraphics[width=\linewidth]{{{stem}.png}}",
            f"    \\caption{{{caption}}}",
            rf"    \label{{fig:{stem}}}",
            r"\end{figure}",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        paths.append(path)
    return paths


def write_p50_error_tolerance_latex_figure(*, out_dir: Path) -> Path:
    paths = write_p50_error_tolerance_latex_figures(out_dir=out_dir, split="test")
    for path in paths:
        if path.name == "rq1_4_1_1_da_price_p50_absolute_error_tolerance_curve.tex":
            return path
    return paths[0]


def write_p50_error_tolerance_summary_latex(summary: pd.DataFrame, *, out_dir: Path, split: str) -> Path | None:
    d = summary.loc[(summary["split"].eq(split)) & (summary["target"].isin(DA_PRICE_TARGETS))].copy()
    if d.empty:
        return None
    order = {label: i for i, label in enumerate(MODEL_LABELS)}
    d["_model_order"] = d["model"].map(order)
    d = d.sort_values("_model_order")
    path = out_dir / "latex" / f"rq1_4_1_1_da_price_p50_error_tolerance_summary_{split}.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \begin{tabular}{@{}lrrrrrr@{}}",
        r"        \toprule",
        r"        \textbf{Model} & \textbf{$\leq$1 €/MWh} & \textbf{$\leq$5 €/MWh} & \textbf{$\leq$10 €/MWh} & \textbf{Median absolute error} & \textbf{MAE p50} & \textbf{N} \\",
        r"        \midrule",
    ]
    for _, row in d.iterrows():
        vals = [
            _latex_escape(row["model"]),
            f"{100.0 * float(row.get('share_le_1', np.nan)):.1f}\\%",
            f"{100.0 * float(row.get('share_le_5', np.nan)):.1f}\\%",
            f"{100.0 * float(row.get('share_le_10', np.nan)):.1f}\\%",
            f"{float(row['median_absolute_error']):.2f}",
            f"{float(row['mae_p50']):.2f}",
            _fmt_int(row["n_obs"]),
        ]
        lines.append("        " + " & ".join(vals) + r" \\")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \caption{DA price p50 absolute error tolerance summary on the test split. Threshold columns report the share of forecasts with absolute p50 error within the stated tolerance.}",
            r"    \label{tab:da_price_p50_error_tolerance_summary_test}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_p50_error_tolerance_performance_values(performance: pd.DataFrame, *, out_dir: Path, split: str) -> list[Path]:
    if performance.empty:
        return []
    csv_dir = out_dir / "csv"
    diag_dir = out_dir / "diagnostics"
    csv_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    d = performance[performance["split"].eq(split)].copy()
    if d.empty:
        return []
    csv_path = csv_dir / f"rq1_4_1_1_price_p50_error_tolerance_performance_values_{split}.csv"
    d.to_csv(csv_path, index=False)
    txt_path = diag_dir / f"rq1_4_1_1_price_p50_error_tolerance_performance_values_{split}.txt"
    lines = [
        f"P50 absolute error tolerance performance values ({split} split)",
        "",
        "Threshold columns report the percentage of observations with absolute p50 error less than or equal to the stated threshold.",
        "The pooled row reports the exact 80th percentile over all price-target common-row absolute p50 errors and all models.",
        "",
    ]
    for _, row in d.iterrows():
        if str(row["target"]) == "ALL_PRICE_TARGETS_POOLED":
            lines.append(f"All price targets / all models: pooled p80 absolute error = {float(row['absolute_error_p80']):.2f} ({row['unit']}); N={int(row['n_obs'])}")
            continue
        cfg = PRICE_P50_TOLERANCE_CONFIG[str(row["target"])]
        shares = []
        for threshold in cfg["thresholds"]:
            slug = _threshold_slug(float(threshold))
            shares.append(f"<={threshold:g} {cfg['unit_latex']}: {float(row[f'share_le_{slug}_pct']):.1f}%")
        lines.append(
            f"{row['target_label']} / {row['model']}: "
            + "; ".join(shares)
            + f"; p80 abs error={float(row['absolute_error_p80']):.2f} {cfg['unit_latex']}; N={int(row['n_obs'])}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, txt_path]


def write_outputs(
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    benchmark_dir: Path,
    out_dir: Path,
    split: str,
    skip_pdf: bool = False,
    p50_tolerance_curve: pd.DataFrame | None = None,
    p50_tolerance_summary: pd.DataFrame | None = None,
    p50_tolerance_performance: pd.DataFrame | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    long_path = csv_dir / "rq1_4_1_1_forecast_metrics_full_long.csv"
    metrics.to_csv(long_path, index=False)
    outputs.append(long_path)

    primary = build_primary_table(metrics, split=split)
    detailed = build_detailed_table(metrics, split=split)

    primary_path = csv_dir / f"rq1_4_1_1_forecast_metrics_full_primary_{split}.csv"
    detailed_path = csv_dir / f"rq1_4_1_1_forecast_metrics_full_detailed_{split}.csv"
    align_path = csv_dir / f"rq1_4_1_1_forecast_metrics_full_alignment_diagnostics_{split}.csv"
    primary.to_csv(primary_path, index=False)
    detailed.to_csv(detailed_path, index=False)
    diagnostics.loc[diagnostics["split"] == split].to_csv(align_path, index=False)
    outputs.extend([primary_path, detailed_path, align_path])

    outputs.extend(write_latex_tables(primary, detailed, out_dir=out_dir, split=split))
    outputs.extend(write_relative_pinball_figure(primary, out_dir=out_dir, split=split, skip_pdf=skip_pdf))
    outputs.extend(write_relative_mae_figure(detailed, out_dir=out_dir, split=split, skip_pdf=skip_pdf))

    if p50_tolerance_curve is not None and not p50_tolerance_curve.empty:
        tolerance_curve_path = csv_dir / "rq1_4_1_1_price_p50_absolute_error_tolerance_curve.csv"
        p50_tolerance_curve.loc[
            (p50_tolerance_curve["split"].eq(split)) & (p50_tolerance_curve["target"].isin(PRICE_P50_TOLERANCE_TARGETS))
        ].to_csv(tolerance_curve_path, index=False)
        outputs.append(tolerance_curve_path)
        outputs.extend(write_p50_error_tolerance_figure(p50_tolerance_curve, out_dir=out_dir, split=split, skip_pdf=skip_pdf))
        outputs.extend(write_p50_error_tolerance_latex_figures(out_dir=out_dir, split=split))
    if p50_tolerance_summary is not None and not p50_tolerance_summary.empty:
        tolerance_summary_path = csv_dir / f"rq1_4_1_1_price_p50_error_tolerance_summary_{split}.csv"
        p50_tolerance_summary.loc[
            (p50_tolerance_summary["split"].eq(split)) & (p50_tolerance_summary["target"].isin(PRICE_P50_TOLERANCE_TARGETS))
        ].to_csv(tolerance_summary_path, index=False)
        outputs.append(tolerance_summary_path)
        tolerance_summary_tex = write_p50_error_tolerance_summary_latex(p50_tolerance_summary, out_dir=out_dir, split=split)
        if tolerance_summary_tex is not None:
            outputs.append(tolerance_summary_tex)
    if p50_tolerance_performance is not None and not p50_tolerance_performance.empty:
        outputs.extend(write_p50_error_tolerance_performance_values(p50_tolerance_performance, out_dir=out_dir, split=split))

    costs = build_computational_cost_table(benchmark_dir)
    if not costs.empty:
        cost_csv = csv_dir / "rq1_4_1_1_computational_cost_365d.csv"
        costs.to_csv(cost_csv, index=False)
        outputs.append(cost_csv)
        cost_table = write_computational_cost_latex_table(costs, out_dir=out_dir)
        if cost_table is not None:
            outputs.append(cost_table)
        outputs.extend(write_computational_cost_figure(costs, out_dir=out_dir, skip_pdf=skip_pdf))

    manifest = {
        "description": "RQ1 4.1.1 full unweighted forecast metrics. Main-text metric is mean pinball loss only.",
        "main_split": split,
        "row_intersection_key": ROW_INTERSECTION_KEY,
        "metrics": METRICS,
        "main_text_metric": "mean_pinball_loss",
        "computational_cost_scaling_days": DEFAULT_TRAINING_DAYS,
        "computational_cost_note": "Thesis-facing computational-cost table and figure report final-training wall-clock only. HPO timing fields remain in the backup CSV for provenance where available.",
        "excluded_from_4_1_1": ["winkler_score", "picp", "pinaw", "coverage", "interval_width", "lead_weighting", "gate_weighting", "tail_weighting"],
        "outputs": [str(p) for p in outputs],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    outputs.append(manifest_path)
    return outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build final RQ1 4.1.1 full unweighted forecast metrics.")
    p.add_argument("--benchmark-root", default="artifacts")
    p.add_argument("--benchmark-dir", action="append", default=[])
    p.add_argument("--out-dir", default="artifacts/benchmark/rq1_ml_model_benchmark/_raw_outputs/4_1_1_full_unweighted")
    p.add_argument("--split", default="test", help="Main thesis/reporting split. Defaults to test.")
    p.add_argument(
        "--splits",
        default="test",
        help="Comma-separated splits to load/export. Defaults to test only; --split selects the main reported split.",
    )
    p.add_argument("--models", default="tft,xgboost,linear", help="Models to compare. Defaults to all RQ1 models: TFT, XGB and RLQR.")
    p.add_argument("--eval-origin-start", default=DEFAULT_EVAL_ORIGIN_START_UTC, help="Inclusive forecast-origin lower bound for final RQ1 evaluation. Empty string disables the lower bound.")
    p.add_argument("--eval-origin-end", default=DEFAULT_EVAL_ORIGIN_END_UTC, help="Inclusive forecast-origin upper bound for final RQ1 evaluation. Empty string disables the upper bound.")
    p.add_argument("--skip-pdf", action="store_true", help="Do not render PDF figures; PNG and LaTeX outputs are still generated.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    models = _parse_models(args.models)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    if args.split not in splits:
        splits.append(args.split)
    benchmark_dirs = _discover_benchmark_dirs(Path(args.benchmark_root), [Path(p) for p in args.benchmark_dir])
    benchmark_dir = benchmark_dirs[0]
    eval_origin_start = _parse_utc_bound(args.eval_origin_start)
    eval_origin_end = _parse_utc_bound(args.eval_origin_end)
    metrics, diagnostics = build_full_metrics(
        benchmark_dir=benchmark_dir,
        models=models,
        splits=splits,
        eval_origin_start=eval_origin_start,
        eval_origin_end=eval_origin_end,
    )
    tolerance_curve, tolerance_summary, tolerance_performance = build_price_p50_error_tolerance_outputs(
        benchmark_dir=benchmark_dir,
        models=models,
        splits=splits,
        eval_origin_start=eval_origin_start,
        eval_origin_end=eval_origin_end,
    )
    outputs = write_outputs(
        metrics,
        diagnostics,
        benchmark_dir=benchmark_dir,
        out_dir=Path(args.out_dir),
        split=args.split,
        skip_pdf=bool(args.skip_pdf),
        p50_tolerance_curve=tolerance_curve,
        p50_tolerance_summary=tolerance_summary,
        p50_tolerance_performance=tolerance_performance,
    )
    print("[OK] Built RQ1 4.1.1 full unweighted forecast metrics.")
    print(f"[OK] benchmark_dir={benchmark_dir}")
    for path in outputs:
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
