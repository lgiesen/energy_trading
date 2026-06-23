#!/usr/bin/env python3
"""Build RQ1 example-week truth-vs-forecast figures for all targets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_trading.visualization.style import THESIS_PALETTE, apply_geo_style, get_model_color, thesis_titlecase
from energy_trading.evaluation.rq1_target_order import model_sort_key, sort_target_frame, target_sort_key


DEFAULT_HIGH_VOLATILITY_START_UTC = "2025-10-05T22:00:00Z"
DEFAULT_TYPICAL_START_UTC = "2025-03-30T22:00:00Z"
DEFAULT_EVAL_ORIGIN_START_UTC = "2025-01-13T23:00:00Z"
DEFAULT_EVAL_ORIGIN_END_UTC = "2026-02-26T21:00:00Z"
DEFAULT_LEAD_H = 24
DEFAULT_QUANTILE = "p50"
DEFAULT_WINDOW_HOURS = 24 * 7
LOCAL_TZ = "Europe/Berlin"
LOCAL_ZONE = ZoneInfo(LOCAL_TZ)
WEEK_TIMEZONE = "UTC"
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TRUTH_COLOR = "#000000"

MODEL_KEYS = {
    "tft": ("tft", "TFT"),
    "xgb": ("xgb", "XGB"),
    "xgboost": ("xgb", "XGB"),
    "linear": ("linear", "RLQR"),
    "rlqr": ("linear", "RLQR"),
}

TARGET_LABELS = {
    "pred_da_price": ("DA", "DA price"),
    "pred_afrr_capacity_price_pos": ("aFRR capacity", "aFRR capacity price +"),
    "pred_afrr_capacity_price_neg": ("aFRR capacity", "aFRR capacity price -"),
    "pred_afrr_activation_price_pos": ("aFRR activation price", "aFRR activation price +"),
    "pred_afrr_activation_price_neg": ("aFRR activation price", "aFRR activation price -"),
    "pred_afrr_activation_rate_pos": ("aFRR activation rate", "aFRR activation rate +"),
    "pred_afrr_activation_rate_neg": ("aFRR activation rate", "aFRR activation rate -"),
}

CANONICAL_TO_PRED_TARGET = {
    "target_da_price": "pred_da_price",
    "target_afrr_capacity_price_pos": "pred_afrr_capacity_price_pos",
    "target_afrr_capacity_price_neg": "pred_afrr_capacity_price_neg",
    "target_afrr_activation_price_vwap_pos": "pred_afrr_activation_price_pos",
    "target_afrr_activation_price_vwap_neg": "pred_afrr_activation_price_neg",
    "target_afrr_activation_rate_pos": "pred_afrr_activation_rate_pos",
    "target_afrr_activation_rate_neg": "pred_afrr_activation_rate_neg",
}

PRED_TO_CANONICAL_TARGET = {v: k for k, v in CANONICAL_TO_PRED_TARGET.items()}

MARKET_ACTIONABLE_TARGET_LABELS = {
    "target_da_price": "DA price",
    "target_afrr_capacity_price_pos": "aFRR capacity price +",
    "target_afrr_capacity_price_neg": "aFRR capacity price -",
    "target_afrr_activation_price_vwap_pos": "aFRR activation price +",
    "target_afrr_activation_price_vwap_neg": "aFRR activation price -",
    "target_afrr_activation_rate_pos": "aFRR activation rate +",
    "target_afrr_activation_rate_neg": "aFRR activation rate -",
}

MARKET_ACTIONABLE_TITLE_LABELS = {
    "target_da_price": "DA Price",
    "target_afrr_capacity_price_pos": "aFRR Capacity Price +",
    "target_afrr_capacity_price_neg": "aFRR Capacity Price -",
    "target_afrr_activation_price_vwap_pos": "aFRR Activation Price +",
    "target_afrr_activation_price_vwap_neg": "aFRR Activation Price -",
    "target_afrr_activation_rate_pos": "aFRR Activation Rate +",
    "target_afrr_activation_rate_neg": "aFRR Activation Rate -",
}

MARKET_ACTIONABLE_CAPTION_LABELS = {
    "target_da_price": "DA price",
    "target_afrr_capacity_price_pos": "aFRR capacity price positive",
    "target_afrr_capacity_price_neg": "aFRR capacity price negative",
    "target_afrr_activation_price_vwap_pos": "aFRR activation price positive",
    "target_afrr_activation_price_vwap_neg": "aFRR activation price negative",
    "target_afrr_activation_rate_pos": "aFRR activation rate positive",
    "target_afrr_activation_rate_neg": "aFRR activation rate negative",
}

MARKET_ACTIONABLE_LABEL_SLUGS = {
    "target_da_price": "da-price",
    "target_afrr_capacity_price_pos": "capacity-price-pos",
    "target_afrr_capacity_price_neg": "capacity-price-neg",
    "target_afrr_activation_price_vwap_pos": "activation-price-pos",
    "target_afrr_activation_price_vwap_neg": "activation-price-neg",
    "target_afrr_activation_rate_pos": "activation-rate-pos",
    "target_afrr_activation_rate_neg": "activation-rate-neg",
}

TARGET_Y_AXIS_LABELS = {
    "target_da_price": "DA price (EUR/MWh)",
    "target_afrr_capacity_price_pos": "Capacity price + (EUR/MW)",
    "target_afrr_capacity_price_neg": "Capacity price - (EUR/MW)",
    "target_afrr_activation_price_vwap_pos": "Activation price + (EUR/MWh)",
    "target_afrr_activation_price_vwap_neg": "Activation price - (EUR/MWh)",
    "target_afrr_activation_rate_pos": "Activation rate + (%)",
    "target_afrr_activation_rate_neg": "Activation rate - (%)",
}

MARKET_ACTIONABLE_SPECS = [
    {
        "market_context": "da_dminus1_11",
        "targets": ["target_da_price"],
        "selection": "local_dminus1_snapshot",
        "forecast_hour": 11,
        "title_prefix": "DA price | D-1 11:00 forecast snapshot | p50",
        "snapshot_description": "D-1 11:00 Europe/Berlin forecast snapshot for next-day DA delivery hours.",
        "filename_prefix": "da_dminus1_11_da_price",
    },
    {
        "market_context": "bcm_dplus1_08",
        "targets": [
            "target_afrr_capacity_price_pos",
            "target_afrr_capacity_price_neg",
            "target_afrr_activation_price_vwap_pos",
            "target_afrr_activation_price_vwap_neg",
            "target_afrr_activation_rate_pos",
            "target_afrr_activation_rate_neg",
        ],
        "selection": "local_dminus1_snapshot",
        "forecast_hour": 8,
        "title_prefix": "BCM D-1 08:00 forecast snapshot",
        "snapshot_description": "BCM D-1 08:00 Europe/Berlin forecast snapshot for next-day capacity bids and activation assumptions.",
        "filename_prefix": "bcm_dplus1_08",
    },
    {
        "market_context": "bem_h1",
        "targets": [
            "target_afrr_activation_price_vwap_pos",
            "target_afrr_activation_price_vwap_neg",
            "target_afrr_activation_rate_pos",
            "target_afrr_activation_rate_neg",
        ],
        "selection": "lead_h1",
        "lead_h": 1.0,
        "title_prefix": "BEM h1",
        "snapshot_description": "BEM h1 forecast for next-hour activation inputs.",
        "filename_prefix": "bem_h1",
    },
]

RESULT_MARKET_ACTIONABLE = {
    (week_key, str(spec["market_context"]), str(target))
    for week_key in ("typical_week", "high_volatility_week")
    for spec in MARKET_ACTIONABLE_SPECS
    for target in spec["targets"]
}

QUANTILE_RE = re.compile(r"^p(\d{1,2})$")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str


@dataclass(frozen=True)
class WeekSpec:
    key: str
    label: str
    start_utc: pd.Timestamp


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
    return sorted(out, key=lambda model: model_sort_key(model.label))


def _parse_quantile(raw: str) -> str:
    q = str(raw).strip().lower()
    if q == "predicted_value":
        return q
    m = QUANTILE_RE.match(q)
    if not m:
        raise ValueError("Quantile must look like p10, p50, p90, or predicted_value.")
    val = int(m.group(1))
    if not 1 <= val <= 99:
        raise ValueError("Quantile pXX must be between p01 and p99.")
    return f"p{val:02d}"


def _parse_timestamp(raw: str) -> pd.Timestamp:
    ts = pd.to_datetime(raw, utc=True, errors="raise")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {raw!r}")
    return pd.Timestamp(ts)


def _parse_optional_timestamp(raw: str | None) -> pd.Timestamp | None:
    if raw is None or str(raw).strip() == "":
        return None
    return _parse_timestamp(str(raw))


def _normalize_forecast_time_column(df: pd.DataFrame) -> pd.DataFrame:
    if "forecast_time_utc" not in df.columns and "snapshot_time_utc" in df.columns:
        return df.rename(columns={"snapshot_time_utc": "forecast_time_utc"})
    return df


def _filter_eval_origin_window(
    df: pd.DataFrame,
    *,
    eval_origin_start: pd.Timestamp | None,
    eval_origin_end: pd.Timestamp | None,
) -> pd.DataFrame:
    if "forecast_time_utc" not in df.columns:
        return df
    out = df.copy()
    out["forecast_time_utc"] = pd.to_datetime(out["forecast_time_utc"], utc=True, errors="coerce")
    if eval_origin_start is not None:
        out = out.loc[out["forecast_time_utc"].ge(eval_origin_start)].copy()
    if eval_origin_end is not None:
        out = out.loc[out["forecast_time_utc"].le(eval_origin_end)].copy()
    return out


def _discover_benchmark_dir(benchmark_root: Path, benchmark_dir: Path | None) -> Path:
    if benchmark_dir is not None:
        out = benchmark_dir.resolve()
    elif (benchmark_root / "diagnostics" / "joined_predictions").exists():
        out = benchmark_root.resolve()
    else:
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
        raise FileNotFoundError(f"Missing joined predictions directory: {joined}")
    return out


def _parse_joined_name(path: Path) -> tuple[str, str, str] | None:
    parts = path.stem.split("__", 2)
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def discover_targets(benchmark_dir: Path, *, split: str, models: list[ModelSpec], requested_targets: list[str] | None) -> list[str]:
    joined = benchmark_dir / "diagnostics" / "joined_predictions"
    available: dict[str, set[str]] = {m.key: set() for m in models}
    for path in joined.glob("*.parquet"):
        parsed = _parse_joined_name(path)
        if parsed is None:
            continue
        model, file_split, target = parsed
        if file_split == split and model in available:
            available[model].add(target)
    if requested_targets:
        targets = requested_targets
    else:
        targets = sorted(set.intersection(*(available[m.key] for m in models)))
    missing = {
        m.key: sorted(set(targets) - available[m.key])
        for m in models
        if set(targets) - available[m.key]
    }
    if missing:
        raise FileNotFoundError(f"Missing joined prediction targets for split={split}: {missing}")
    if not targets:
        raise FileNotFoundError(f"No common targets found for split={split} in {joined}.")
    return targets


def build_week_specs(args: argparse.Namespace) -> list[WeekSpec]:
    if args.date:
        return [WeekSpec("custom", "Custom", _parse_timestamp(args.date))]
    return [
        WeekSpec("high_volatility", "High volatility", _parse_timestamp(args.high_volatility_start)),
        WeekSpec("typical", "Typical", _parse_timestamp(args.typical_start)),
    ]


def _read_model_target(
    *,
    benchmark_dir: Path,
    model: ModelSpec,
    split: str,
    target: str,
    lead_h: float,
    quantile: str,
    eval_origin_start: pd.Timestamp | None = None,
    eval_origin_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    path = benchmark_dir / "diagnostics" / "joined_predictions" / f"{model.key}__{split}__{target}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing joined prediction file: {path}")
    df = _normalize_forecast_time_column(pd.read_parquet(path))
    required = {"target_time_utc", "lead_time_h", "y_true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if quantile not in df.columns:
        if quantile == "p50" and "predicted_value" in df.columns:
            pred_col = "predicted_value"
        else:
            raise ValueError(f"{path} does not contain requested quantile column {quantile!r}.")
    else:
        pred_col = quantile
    optional_time_cols = ["forecast_time_utc"] if "forecast_time_utc" in df.columns else []
    out = df[[*optional_time_cols, "target_time_utc", "lead_time_h", "y_true", pred_col]].copy()
    out = _filter_eval_origin_window(out, eval_origin_start=eval_origin_start, eval_origin_end=eval_origin_end)
    out["target_time_utc"] = pd.to_datetime(out["target_time_utc"], utc=True, errors="coerce")
    out["lead_time_h"] = pd.to_numeric(out["lead_time_h"], errors="coerce")
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out["y_pred"] = pd.to_numeric(out[pred_col], errors="coerce")
    out = out.loc[np.isclose(out["lead_time_h"], float(lead_h), atol=1e-9)].copy()
    out = out.dropna(subset=["target_time_utc", "y_true", "y_pred"])
    out = out.sort_values("target_time_utc").drop_duplicates("target_time_utc", keep="last")
    out = out[["target_time_utc", "y_true", "y_pred"]]
    out = out.rename(columns={"y_pred": f"{model.key}_{quantile}"})
    return out


def load_merged_target(
    *,
    benchmark_dir: Path,
    models: list[ModelSpec],
    split: str,
    target: str,
    lead_h: float,
    quantile: str,
    eval_origin_start: pd.Timestamp | None = None,
    eval_origin_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for model in models:
        part = _read_model_target(
            benchmark_dir=benchmark_dir,
            model=model,
            split=split,
            target=target,
            lead_h=lead_h,
            quantile=quantile,
            eval_origin_start=eval_origin_start,
            eval_origin_end=eval_origin_end,
        )
        if merged is None:
            merged = part.rename(columns={f"{model.key}_{quantile}": f"{model.key}_pred"})
        else:
            part = part.drop(columns=["y_true"]).rename(columns={f"{model.key}_{quantile}": f"{model.key}_pred"})
            merged = merged.merge(part, on="target_time_utc", how="inner")
    if merged is None or merged.empty:
        raise ValueError(f"No merged rows for split={split}, target={target}, lead={lead_h}, quantile={quantile}.")
    return merged.sort_values("target_time_utc").reset_index(drop=True)


def _canonical_target(target: str) -> str:
    return PRED_TO_CANONICAL_TARGET.get(str(target), str(target))


def _prediction_target(canonical_target: str) -> str:
    return CANONICAL_TO_PRED_TARGET.get(str(canonical_target), str(canonical_target))


def _target_info(target: str) -> tuple[str, str]:
    return TARGET_LABELS.get(target, ("Other", target.replace("_", " ")))


def _market_target_label(canonical_target: str) -> str:
    return MARKET_ACTIONABLE_TARGET_LABELS.get(str(canonical_target), str(canonical_target).replace("_", " "))


def target_y_axis_label(target: str, target_display: str | None = None) -> str:
    canonical_target = _canonical_target(str(target))
    return TARGET_Y_AXIS_LABELS.get(canonical_target, str(target_display or _market_target_label(canonical_target)))


def target_y_axis_unit_label(target: str) -> str:
    canonical_target = _canonical_target(str(target))
    return target_y_axis_label(canonical_target)


def _is_activation_rate_target(target: str) -> bool:
    return "activation_rate" in _canonical_target(str(target))


def _plot_y_scale(target: str) -> float:
    return 100.0 if _is_activation_rate_target(target) else 1.0


def _scale_plot_frame_for_target(frame: pd.DataFrame, *, target: str, columns: list[str]) -> pd.DataFrame:
    scale = _plot_y_scale(target)
    if abs(scale - 1.0) <= 1e-12:
        return frame
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * scale
    return out


def _window_slice(df: pd.DataFrame, start_utc: pd.Timestamp, window_hours: int) -> pd.DataFrame:
    end_utc = start_utc + pd.Timedelta(hours=int(window_hours))
    ts = pd.to_datetime(df["target_time_utc"], utc=True, errors="coerce")
    return df.loc[(ts >= start_utc) & (ts < end_utc)].copy()


def _week_start_utc(times: pd.Series) -> pd.Series:
    ts = pd.to_datetime(times, utc=True, errors="coerce")
    return ts.dt.tz_convert(None).dt.to_period("W").dt.start_time.dt.tz_localize("UTC")


def _week_end_utc(week_start: Any) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(week_start, utc=True)) + pd.Timedelta(days=7)


def _read_week_selection_base(
    *,
    benchmark_dir: Path,
    model: ModelSpec,
    split: str,
    canonical_target: str,
    eval_origin_start: pd.Timestamp | None = None,
    eval_origin_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    target = _prediction_target(canonical_target)
    path = benchmark_dir / "diagnostics" / "joined_predictions" / f"{model.key}__{split}__{target}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing joined prediction file for week selection: {path}")
    df = _normalize_forecast_time_column(pd.read_parquet(path))
    required = {"target_time_utc", "y_true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required week-selection columns: {sorted(missing)}")
    optional_time_cols = ["forecast_time_utc"] if "forecast_time_utc" in df.columns else []
    out = df[[*optional_time_cols, "target_time_utc", "y_true"]].copy()
    out = _filter_eval_origin_window(out, eval_origin_start=eval_origin_start, eval_origin_end=eval_origin_end)
    out["target_time_utc"] = pd.to_datetime(out["target_time_utc"], utc=True, errors="coerce")
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out = out.dropna(subset=["target_time_utc", "y_true"]).copy()
    out["split"] = split
    out["target"] = canonical_target
    out["target_display"] = _market_target_label(canonical_target)
    out["target_group"] = _target_info(_prediction_target(canonical_target))[0]
    out["week_start_utc"] = _week_start_utc(out["target_time_utc"])
    return out


def _weekly_target_stats(base: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        base.groupby(["split", "target", "target_display", "target_group", "week_start_utc"], as_index=False)
        .agg(weekly_std=("y_true", "std"), n_rows=("y_true", "size"))
    )
    if weekly.empty:
        return weekly
    inferred_expected = weekly.groupby(["split", "target"], as_index=False)["n_rows"].max().rename(columns={"n_rows": "expected_n_rows"})
    weekly = weekly.merge(inferred_expected, on=["split", "target"], how="left")
    weekly["completeness"] = weekly["n_rows"] / weekly["expected_n_rows"].replace(0, np.nan)
    weekly["week_end_utc"] = pd.to_datetime(weekly["week_start_utc"], utc=True) + pd.Timedelta(days=7)
    weekly["week_timezone"] = WEEK_TIMEZONE
    return weekly


def select_typical_weeks_algorithmic(base: pd.DataFrame, *, min_completeness: float = 0.80) -> pd.DataFrame:
    weekly = _weekly_target_stats(base)
    rows: list[pd.Series] = []
    for (split, target), part in weekly.groupby(["split", "target"], sort=False):
        usable = part.loc[part["completeness"].ge(float(min_completeness)) | part["completeness"].isna()].copy()
        if usable.empty:
            usable = part.copy()
        median_std = float(pd.to_numeric(usable["weekly_std"], errors="coerce").median())
        usable["median_weekly_std"] = median_std
        usable["abs_distance_to_median_std"] = (pd.to_numeric(usable["weekly_std"], errors="coerce") - median_std).abs()
        usable = usable.sort_values(["abs_distance_to_median_std", "n_rows", "week_start_utc"], ascending=[True, False, True])
        if usable.empty:
            continue
        row = usable.iloc[0].copy()
        row["selection_type"] = "typical_week"
        row["selection_rank"] = 1
        row["selection_scope"] = "target"
        row["selection_rule"] = "Selected test-set UTC week whose weekly std(y_true) is closest to the median weekly std(y_true), separately per target."
        row["source"] = "computed_in_example_week_script"
        row["volatility_score"] = float(row["weekly_std"]) if pd.notna(row["weekly_std"]) else np.nan
        row["week_timezone"] = WEEK_TIMEZONE
        rows.append(row)
    return pd.DataFrame(rows)


def select_high_volatility_weeks_algorithmic(base: pd.DataFrame, *, min_completeness: float = 0.80) -> pd.DataFrame:
    weekly = _weekly_target_stats(base)
    rows: list[pd.Series] = []
    for (split, target), part in weekly.groupby(["split", "target"], sort=False):
        usable = part.loc[part["completeness"].ge(float(min_completeness)) | part["completeness"].isna()].copy()
        if usable.empty:
            usable = part.copy()
        usable["volatility_score"] = pd.to_numeric(usable["weekly_std"], errors="coerce")
        usable = usable.sort_values(["volatility_score", "n_rows", "week_start_utc"], ascending=[False, False, True])
        if usable.empty:
            continue
        row = usable.iloc[0].copy()
        row["selection_type"] = "high_volatility_week"
        row["selection_rank"] = 1
        row["selection_scope"] = "target"
        row["selection_rule"] = "Selected highest weekly std(y_true) test-set UTC week, separately per target."
        row["source"] = "computed_in_example_week_script"
        row["median_weekly_std"] = np.nan
        row["abs_distance_to_median_std"] = np.nan
        row["week_timezone"] = WEEK_TIMEZONE
        rows.append(row)
    return pd.DataFrame(rows)


def _find_tail_spike_selected_weeks(out_dir: Path, explicit: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            out_dir.parent / "4_1_5_tail_spike" / "tail_spike_selected_weeks.csv",
            out_dir.parent / "4_1_5_tail_spike" / "backup" / "diagnostics" / "tail_spike_selected_weeks.csv",
            out_dir.parent.parent / "4_1_5_tail_spike" / "backup" / "diagnostics" / "tail_spike_selected_weeks.csv",
            Path("artifacts/benchmark/rq1_ml_model_benchmark/4_1_5_tail_spike/backup/diagnostics/tail_spike_selected_weeks.csv"),
            Path("artifacts/benchmark/rq1_ml_model_benchmark/_raw_outputs/4_1_5_tail_spike/tail_spike_selected_weeks.csv"),
            Path("artifacts/rq1_ml_model_benchmark/4_1_5_tail_spike/backup/diagnostics/tail_spike_selected_weeks.csv"),
        ]
    )
    return next((p for p in candidates if p.exists()), None)


def _normalize_selected_week_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["week_start_utc"] = pd.to_datetime(out["week_start_utc"], utc=True, errors="coerce")
    if "week_end_utc" not in out.columns:
        out["week_end_utc"] = out["week_start_utc"] + pd.Timedelta(days=7)
    else:
        out["week_end_utc"] = pd.to_datetime(out["week_end_utc"], utc=True, errors="coerce")
    for col in ["weekly_std", "median_weekly_std", "abs_distance_to_median_std", "volatility_score", "n_rows", "selection_rank"]:
        if col not in out.columns:
            out[col] = np.nan
    for col in ["split", "target", "target_display", "target_group", "selection_type", "selection_scope", "selection_rule", "source"]:
        if col not in out.columns:
            out[col] = ""
    if "week_timezone" not in out.columns:
        out["week_timezone"] = WEEK_TIMEZONE
    out["week_timezone"] = out["week_timezone"].replace("", WEEK_TIMEZONE).fillna(WEEK_TIMEZONE)
    return out


def _prepare_tail_spike_high_vol_source(
    src: pd.DataFrame,
    *,
    split: str,
    canonical_targets: list[str],
    source: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    if "target" not in src.columns or src["target"].isna().all():
        warnings.append(
            {
                "selection_mode": "algorithmic",
                "split": split,
                "target": "",
                "warning": "tail_spike_selected_weeks_is_not_target_specific_recomputed_high_volatility",
                "source": str(source),
            }
        )
        return pd.DataFrame(), warnings
    if "target_group" in src.columns and not set(src["target"].dropna().astype(str)).intersection(set(canonical_targets)):
        warnings.append(
            {
                "selection_mode": "algorithmic",
                "split": split,
                "target": "",
                "warning": "tail_spike_selected_weeks_uses_target_group_only_recomputed_high_volatility",
                "source": str(source),
            }
        )
        return pd.DataFrame(), warnings
    if "split" not in src.columns:
        warnings.append({"selection_mode": "algorithmic", "split": split, "target": "", "warning": "tail_spike_selected_weeks_missing_split_recomputed_high_volatility", "source": str(source)})
        return pd.DataFrame(), warnings
    if "selection_type" not in src.columns:
        warnings.append({"selection_mode": "algorithmic", "split": split, "target": "", "warning": "tail_spike_selected_weeks_missing_selection_type_recomputed_high_volatility", "source": str(source)})
        return pd.DataFrame(), warnings

    out = src.loc[
        src["split"].astype(str).eq(split)
        & src["selection_type"].astype(str).eq("high_volatility_week")
        & src["target"].astype(str).isin(canonical_targets)
    ].copy()
    if out.empty:
        return out, warnings

    if "week_start_utc" not in out.columns:
        warnings.append({"selection_mode": "algorithmic", "split": split, "target": "", "warning": "tail_spike_selected_weeks_missing_week_start_recomputed_high_volatility", "source": str(source)})
        return pd.DataFrame(), warnings
    out["week_start_utc"] = pd.to_datetime(out["week_start_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["week_start_utc"]).copy()
    if out.empty:
        return out, warnings

    if "week_end_utc" not in out.columns:
        out["week_end_utc"] = out["week_start_utc"] + pd.Timedelta(days=7)
    else:
        out["week_end_utc"] = pd.to_datetime(out["week_end_utc"], utc=True, errors="coerce")
        out["week_end_utc"] = out["week_end_utc"].fillna(out["week_start_utc"] + pd.Timedelta(days=7))
    if "week_timezone" not in out.columns:
        out["week_timezone"] = WEEK_TIMEZONE
    out["week_timezone"] = out["week_timezone"].replace("", WEEK_TIMEZONE).fillna(WEEK_TIMEZONE)

    if "volatility_score" in out.columns:
        out["volatility_score"] = pd.to_numeric(out["volatility_score"], errors="coerce")
    elif "weekly_std" in out.columns:
        out["volatility_score"] = pd.to_numeric(out["weekly_std"], errors="coerce")
    else:
        warnings.append({"selection_mode": "algorithmic", "split": split, "target": "", "warning": "tail_spike_selected_weeks_missing_volatility_score_and_weekly_std_recomputed_high_volatility", "source": str(source)})
        return pd.DataFrame(), warnings
    if out["volatility_score"].isna().all():
        warnings.append({"selection_mode": "algorithmic", "split": split, "target": "", "warning": "tail_spike_selected_weeks_non_numeric_volatility_score_recomputed_high_volatility", "source": str(source)})
        return pd.DataFrame(), warnings

    if "weekly_std" not in out.columns:
        out["weekly_std"] = out["volatility_score"]
    out["weekly_std"] = pd.to_numeric(out["weekly_std"], errors="coerce").fillna(out["volatility_score"])
    if "n_rows" not in out.columns:
        out["n_rows"] = np.nan
    out["n_rows"] = pd.to_numeric(out["n_rows"], errors="coerce")
    if "selection_rank" not in out.columns or pd.to_numeric(out.get("selection_rank"), errors="coerce").isna().all():
        out = out.sort_values(["target", "volatility_score", "n_rows", "week_start_utc"], ascending=[True, False, False, True]).copy()
        out["selection_rank"] = out.groupby("target").cumcount() + 1
    out["selection_rank"] = pd.to_numeric(out["selection_rank"], errors="coerce")
    for col, default in {
        "target_display": "",
        "target_group": "",
        "selection_scope": "target",
        "selection_rule": "Selected highest weekly std(y_true) test-set UTC week from tail/spike selected-week diagnostics, separately per target.",
        "source": "tail_spike_selected_weeks.csv",
        "median_weekly_std": np.nan,
        "abs_distance_to_median_std": np.nan,
    }.items():
        if col not in out.columns:
            out[col] = default
    out["source"] = "tail_spike_selected_weeks.csv"
    for target in canonical_targets:
        mask = out["target"].astype(str).eq(target)
        if not mask.any():
            continue
        out.loc[mask & out["target_display"].astype(str).eq(""), "target_display"] = _market_target_label(target)
        out.loc[mask & out["target_group"].astype(str).eq(""), "target_group"] = _target_info(_prediction_target(target))[0]
    out["selection_rule"] = out["selection_rule"].replace("", "Selected highest weekly std(y_true) test-set UTC week from tail/spike selected-week diagnostics, separately per target.").fillna(
        "Selected highest weekly std(y_true) test-set UTC week from tail/spike selected-week diagnostics, separately per target."
    )
    return _normalize_selected_week_columns(out), warnings


def build_algorithmic_selected_weeks(
    *,
    benchmark_dir: Path,
    out_dir: Path,
    models: list[ModelSpec],
    split: str,
    canonical_targets: list[str],
    tail_spike_selected_weeks: str | None = None,
    eval_origin_start: pd.Timestamp | None = None,
    eval_origin_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split != "test":
        raise ValueError(f"Algorithmic final example-week mode only supports test split by default; got split={split!r}.")
    base_parts = [
        _read_week_selection_base(
            benchmark_dir=benchmark_dir,
            model=models[0],
            split=split,
            canonical_target=target,
            eval_origin_start=eval_origin_start,
            eval_origin_end=eval_origin_end,
        )
        for target in canonical_targets
    ]
    base = pd.concat(base_parts, ignore_index=True) if base_parts else pd.DataFrame()
    typical = select_typical_weeks_algorithmic(base)
    hv_computed = select_high_volatility_weeks_algorithmic(base)
    warnings: list[dict[str, Any]] = []
    source = _find_tail_spike_selected_weeks(out_dir, explicit=tail_spike_selected_weeks)
    hv_source_rows: list[pd.DataFrame] = []
    if source is not None:
        src, source_warnings = _prepare_tail_spike_high_vol_source(
            pd.read_csv(source),
            split=split,
            canonical_targets=canonical_targets,
            source=source,
        )
        warnings.extend(source_warnings)
        if not src.empty:
            src["target"] = src["target"].astype(str)
            for target in canonical_targets:
                part = src.loc[src["target"].eq(target)].copy()
                if part.empty:
                    warnings.append({"selection_mode": "algorithmic", "split": split, "target": target, "warning": "missing_tail_spike_selected_week_for_target", "source": str(source)})
                    continue
                part = part.sort_values(["selection_rank", "volatility_score", "n_rows", "week_start_utc"], ascending=[True, False, False, True]).head(1).copy()
                part["source"] = "tail_spike_selected_weeks.csv"
                hv_source_rows.append(part)
    else:
        warnings.append({"selection_mode": "algorithmic", "split": split, "target": "", "warning": "missing_tail_spike_selected_weeks_source", "source": ""})
    hv_from_source = pd.concat(hv_source_rows, ignore_index=True) if hv_source_rows else pd.DataFrame()
    missing_targets = set(canonical_targets) - set(hv_from_source["target"].astype(str)) if not hv_from_source.empty else set(canonical_targets)
    hv_fallback = hv_computed.loc[hv_computed["target"].isin(missing_targets)].copy()
    if not hv_fallback.empty:
        for target in sorted(set(hv_fallback["target"].astype(str))):
            warnings.append({"selection_mode": "algorithmic", "split": split, "target": target, "warning": "fallback_to_computed_high_volatility_week", "source": "computed_in_example_week_script"})
    selected = pd.concat([typical, hv_from_source, hv_fallback], ignore_index=True, sort=False)
    selected = _normalize_selected_week_columns(selected)
    selected["selection_mode"] = "algorithmic"
    if not selected["split"].astype(str).eq(split).all():
        raise ValueError("Algorithmic selected weeks contain non-test/non-requested split rows.")
    warning_df = pd.DataFrame(warnings)
    return selected, warning_df


def _read_market_actionable_model_target(
    *,
    benchmark_dir: Path,
    model: ModelSpec,
    split: str,
    canonical_target: str,
    eval_origin_start: pd.Timestamp | None = None,
    eval_origin_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    target = _prediction_target(canonical_target)
    path = benchmark_dir / "diagnostics" / "joined_predictions" / f"{model.key}__{split}__{target}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing joined prediction file: {path}")
    df = _normalize_forecast_time_column(pd.read_parquet(path))
    required = {"forecast_time_utc", "target_time_utc", "lead_time_h", "y_true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns for market-actionable example weeks: {sorted(missing)}. "
            "Cannot infer DA/BCM market snapshots without forecast_time_utc."
        )
    pred_col = "p50" if "p50" in df.columns else "predicted_value"
    if pred_col not in df.columns:
        raise ValueError(f"{path} does not contain p50 or predicted_value.")
    optional_quantiles = [q for q in ["p10", "p90"] if q in df.columns]
    out = df[["forecast_time_utc", "target_time_utc", "lead_time_h", "y_true", pred_col, *optional_quantiles]].copy()
    out["forecast_time_utc"] = pd.to_datetime(out["forecast_time_utc"], utc=True, errors="coerce")
    out["target_time_utc"] = pd.to_datetime(out["target_time_utc"], utc=True, errors="coerce")
    out = _filter_eval_origin_window(out, eval_origin_start=eval_origin_start, eval_origin_end=eval_origin_end)
    out["lead_time_h"] = pd.to_numeric(out["lead_time_h"], errors="coerce")
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out["p50"] = pd.to_numeric(out[pred_col], errors="coerce")
    for q in ["p10", "p90"]:
        out[q] = pd.to_numeric(out[q], errors="coerce") if q in out.columns else np.nan
    out = out.dropna(subset=["forecast_time_utc", "target_time_utc", "lead_time_h", "y_true", "p50"]).copy()
    out["model"] = model.key
    out["model_label"] = model.label
    out["target"] = canonical_target
    out["target_display"] = _market_target_label(canonical_target)
    return out[["model", "model_label", "target", "target_display", "forecast_time_utc", "target_time_utc", "lead_time_h", "y_true", "p10", "p50", "p90"]]


def _select_market_actionable_rows(
    df: pd.DataFrame,
    *,
    spec: dict[str, Any],
    week: WeekSpec,
    window_hours: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    d = _window_slice(df, week.start_utc, window_hours)
    warning = ""
    if d.empty:
        warning = "No target timestamps in selected week."
        selected = d
    elif spec["selection"] == "lead_h1":
        lead_h = float(spec["lead_h"])
        selected = d.loc[np.isclose(pd.to_numeric(d["lead_time_h"], errors="coerce"), lead_h, atol=1e-9)].copy()
        if selected.empty:
            warning = "No h1 rows available for selected week."
    elif spec["selection"] == "local_dminus1_snapshot":
        forecast_local = pd.to_datetime(d["forecast_time_utc"], utc=True).dt.tz_convert(LOCAL_ZONE)
        target_local = pd.to_datetime(d["target_time_utc"], utc=True).dt.tz_convert(LOCAL_ZONE)
        mask = (
            forecast_local.dt.hour.eq(int(spec["forecast_hour"]))
            & forecast_local.dt.minute.eq(0)
            & target_local.dt.normalize().eq(forecast_local.dt.normalize() + pd.Timedelta(days=1))
            & target_local.dt.hour.between(0, 23)
        )
        selected = d.loc[mask].copy()
        if selected.empty:
            warning = f"No D-1 {int(spec['forecast_hour']):02d}:00 Europe/Berlin snapshot rows available for selected week."
    else:
        raise ValueError(f"Unknown market-actionable selection: {spec['selection']}")

    if not selected.empty:
        selected = selected.sort_values(["target_time_utc", "model"]).drop_duplicates(["target_time_utc", "model"], keep="last")
    observed_forecast_times = ""
    observed_leads = ""
    if not selected.empty:
        observed_forecast_times = "; ".join(
            sorted(
                pd.to_datetime(selected["forecast_time_utc"], utc=True)
                .dt.tz_convert(LOCAL_ZONE)
                .dt.strftime("%Y-%m-%d %H:%M")
                .unique()
                .tolist()
            )
        )
        observed_leads = ",".join(
            str(int(x)) if float(x).is_integer() else f"{float(x):g}"
            for x in sorted(pd.to_numeric(selected["lead_time_h"], errors="coerce").dropna().unique())
        )
    diagnostic = {
        "selection_rule": str(spec["selection"]),
        "expected_snapshot": str(spec["snapshot_description"]),
        "observed_forecast_times_local": observed_forecast_times,
        "observed_leads": observed_leads,
        "n_rows": int(len(selected)),
        "warning": warning,
    }
    return selected, diagnostic


def _market_actionable_filename(spec: dict[str, Any], canonical_target: str, week: WeekSpec) -> str:
    target_slug = {
        "target_da_price": "da_price",
        "target_afrr_capacity_price_pos": "capacity_price_pos",
        "target_afrr_capacity_price_neg": "capacity_price_neg",
        "target_afrr_activation_price_vwap_pos": "activation_price_pos",
        "target_afrr_activation_price_vwap_neg": "activation_price_neg",
        "target_afrr_activation_rate_pos": "activation_rate_pos",
        "target_afrr_activation_rate_neg": "activation_rate_neg",
    }.get(canonical_target, _safe_slug(canonical_target))
    prefix = str(spec["filename_prefix"])
    if prefix.endswith(target_slug):
        return f"{prefix}_{week.key}.png"
    return f"{prefix}_{target_slug}_{week.key}.png"


def _market_actionable_title(spec: dict[str, Any], target_display: str) -> str:
    del spec
    return f"{target_display}: p50 Forecast"


def _week_type_label(week: WeekSpec) -> str:
    if week.key == "typical_week":
        return "Typical week"
    if week.key == "high_volatility_week":
        return "High-volatility week"
    if week.key == "typical":
        return "Typical week"
    if week.key in {"high_volatility", "high-volatility"}:
        return "High-volatility week"
    return f"{week.label} week".strip()


def _market_actionable_target_title(canonical_target: str) -> str:
    return MARKET_ACTIONABLE_TITLE_LABELS.get(canonical_target, _market_target_label(canonical_target))


def _market_actionable_caption_target(canonical_target: str) -> str:
    return MARKET_ACTIONABLE_CAPTION_LABELS.get(canonical_target, _market_target_label(canonical_target))


def _is_activation_assumption_target(canonical_target: str) -> bool:
    return "activation_price" in canonical_target or "activation_rate" in canonical_target


def _market_snapshot_label(spec: dict[str, Any], canonical_target: str, *, latex: bool = False) -> str:
    context = str(spec["market_context"])
    dminus = "D$-1$" if latex else "D-1"
    if context == "da_dminus1_11":
        return f"DA {dminus} 11:00 Europe/Berlin forecast snapshot"
    if context == "bcm_dplus1_08":
        if _is_activation_assumption_target(canonical_target):
            if latex:
                return f"BCM {dminus} 08:00 Europe/Berlin activation-assumption forecast snapshot"
            return f"BCM {dminus} 08:00 Europe/Berlin activation assumption"
        return f"BCM {dminus} 08:00 Europe/Berlin forecast snapshot"
    if context == "bem_h1":
        return "BEM h1 forecast"
    return str(spec.get("snapshot_description", "")).strip()


def _market_actionable_subtitle(week: WeekSpec, spec: dict[str, Any], canonical_target: str) -> str:
    return f"{_week_type_label(week)} | {_market_snapshot_label(spec, canonical_target, latex=False)}"


def _market_actionable_short_caption(canonical_target: str) -> str:
    return f"Example-week {_market_actionable_caption_target(canonical_target)} forecasts"


def _market_actionable_caption(week: WeekSpec, spec: dict[str, Any], canonical_target: str) -> str:
    selection_label = _week_type_label(week).lower()
    return (
        f"Example-week {_market_actionable_caption_target(canonical_target)} forecasts for the algorithmically selected {selection_label} using the "
        f"{_market_snapshot_label(spec, canonical_target, latex=True)}. The figure compares realized values with p50 forecasts from RLQR, XGB and TFT."
    )


def _market_actionable_figure_label(week: WeekSpec, spec: dict[str, Any], canonical_target: str) -> str:
    week_slug = _safe_slug(_week_type_label(week).replace(" week", "").replace("-", " "))
    target_slug = MARKET_ACTIONABLE_LABEL_SLUGS.get(canonical_target, _safe_slug(canonical_target).replace("_", "-"))
    context = str(spec["market_context"])
    context_suffix = ""
    if context == "bem_h1":
        context_suffix = "-bem"
    elif context == "bcm_dplus1_08" and _is_activation_assumption_target(canonical_target):
        context_suffix = "-bcm"
    return f"fig:rq1-example-week-{target_slug}{context_suffix}-{week_slug}"


def _plot_market_actionable(
    *,
    selected: pd.DataFrame,
    spec: dict[str, Any],
    week: WeekSpec,
    canonical_target: str,
    models: list[ModelSpec],
    out_path: Path,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    if selected.empty:
        raise ValueError(f"Empty market-actionable selection for {week.key}/{spec['market_context']}/{canonical_target}.")
    pivot = selected.pivot_table(
        index="target_time_utc",
        columns="model",
        values="p50",
        aggfunc="first",
    ).sort_index()
    truth = selected[["target_time_utc", "y_true"]].drop_duplicates("target_time_utc").set_index("target_time_utc").sort_index()
    d = truth.join(pivot, how="inner").reset_index()
    d["local_time"] = pd.to_datetime(d["target_time_utc"], utc=True).dt.tz_convert(LOCAL_ZONE)
    d["plot_idx"] = np.arange(len(d), dtype=int)
    d = _scale_plot_frame_for_target(d, target=canonical_target, columns=["y_true", *[model.key for model in models]])
    target_display = _market_target_label(canonical_target)
    figure_title = _market_actionable_title(spec, _market_actionable_target_title(canonical_target))
    figure_subtitle = _market_actionable_subtitle(week, spec, canonical_target)
    caption = _market_actionable_caption(week, spec, canonical_target)
    short_caption = _market_actionable_short_caption(canonical_target)
    y_axis_label = target_y_axis_label(canonical_target, target_display)
    apply_geo_style()
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    ax.plot(d["plot_idx"], d["y_true"], label="Truth", color=TRUTH_COLOR, linewidth=2.3)
    row: dict[str, Any] = {
        "week_type": week.key,
        "market_context": spec["market_context"],
        "target": canonical_target,
        "target_display": target_display,
        "y_axis_label": y_axis_label,
        "figure_path": str(out_path),
        "figure_title": figure_title,
        "figure_subtitle": figure_subtitle,
        "caption": caption,
        "short_caption": short_caption,
        "market_context_label": _market_snapshot_label(spec, canonical_target, latex=False),
    }
    for model in models:
        if model.key not in d.columns:
            continue
        mae = float(np.mean(np.abs(pd.to_numeric(d[model.key], errors="coerce") - pd.to_numeric(d["y_true"], errors="coerce"))))
        ax.plot(d["plot_idx"], d[model.key], label=f"{model.label} p50 (MAE={mae:.3f})", color=get_model_color(model.key), linewidth=1.8)
        row[f"mae_{model.key}"] = mae
    fig.suptitle(figure_title, y=0.965)
    ax.set_title(figure_subtitle, fontsize=10, pad=3)
    ax.set_xlabel("Time")
    ax.set_ylabel(y_axis_label)
    ax.set_xlim(*_plot_index_xlim(d))
    ax.margins(x=0)
    xticks, xticklabels = _latex_week_ticks(d, week)
    tick_positions = [int(x) for x in xticks.split(",") if x] if xticks else []
    tick_labels = xticklabels.split(",") if xticklabels else []
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis="x", rotation=0)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.90))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return row


def _market_actionable_frame(selected: pd.DataFrame) -> pd.DataFrame:
    pivot = selected.pivot_table(
        index="target_time_utc",
        columns="model",
        values="p50",
        aggfunc="first",
    ).sort_index()
    truth = selected[["target_time_utc", "y_true"]].drop_duplicates("target_time_utc").set_index("target_time_utc").sort_index()
    d = truth.join(pivot, how="inner").reset_index()
    d["local_time"] = pd.to_datetime(d["target_time_utc"], utc=True).dt.tz_convert(LOCAL_ZONE)
    d["plot_idx"] = np.arange(len(d), dtype=int)
    return d


def _write_market_actionable_latex(
    *,
    selected: pd.DataFrame,
    spec: dict[str, Any],
    week: WeekSpec,
    canonical_target: str,
    models: list[ModelSpec],
    tex_path: Path,
    caption: str,
    label: str,
) -> Path:
    d = _market_actionable_frame(selected)
    if d.empty:
        raise ValueError(f"Empty market-actionable LaTeX data for {week.key}/{spec['market_context']}/{canonical_target}.")
    d = _scale_plot_frame_for_target(d, target=canonical_target, columns=["y_true", *[model.key for model in models]])
    date_range = _week_date_range_label(d)
    xticks, xticklabels = _latex_week_ticks(d, week)
    xmin, xmax = _plot_index_xlim(d)
    y_values = [pd.to_numeric(d["y_true"], errors="coerce").to_numpy(dtype=float)]
    for model in models:
        if model.key in d.columns:
            y_values.append(pd.to_numeric(d[model.key], errors="coerce").to_numpy(dtype=float))
    ylim = _padded_ylim(np.concatenate(y_values)) if y_values else None
    color_lines = [
        f"\\definecolor{{{_latex_color_name(role)}}}{{HTML}}{{{hex_color.lstrip('#').upper()}}}"
        for role, hex_color in THESIS_PALETTE.items()
    ]
    lines = [
        r"% Requires: \usepackage{pgfplots}",
        r"% Requires: \usepackage{xcolor}",
        r"% Recommended in preamble: \pgfplotsset{compat=1.18}",
        *color_lines,
        r"\begin{figure}[htbp]",
        r"    \centering",
        r"    \resizebox{\linewidth}{!}{%",
        r"        \begin{tikzpicture}",
        r"            \begin{axis}[",
        r"                width=0.98\textwidth,",
        r"                height=7cm,",
        rf"                title={{{_latex_escape(_market_actionable_title(spec, _market_actionable_target_title(canonical_target)))}}},",
        r"                title style={at={(0.5,1.16)}, anchor=south},",
        r"                xlabel={Time},",
        rf"                ylabel={{{_latex_escape(target_y_axis_unit_label(canonical_target))}}},",
        r"                legend style={at={(0.5,1.06)}, anchor=south, legend columns=-1, draw=none, fill=none, text=black},",
        r"                legend cell align={left},",
        r"                axis lines*=left,",
        r"                grid=major,",
        rf"                xmin={_tex_num(xmin)},",
        rf"                xmax={_tex_num(xmax)},",
        r"                enlarge x limits=false,",
        r"                clip mode=individual,",
    ]
    if xticks and xticklabels:
        lines.extend(
            [
                rf"                xtick={{{xticks}}},",
                rf"                xticklabels={{{xticklabels}}},",
                r"                xticklabel style={align=center, rotate=0},",
            ]
        )
    if ylim is not None:
        lines.extend(
            [
                rf"                ymin={_tex_num(ylim[0])},",
                rf"                ymax={_tex_num(ylim[1])},",
            ]
        )
    lines.append(r"            ]")
    truth_coords = " ".join(f"({_tex_num(i)},{_tex_num(y)})" for i, y in zip(d["plot_idx"], d["y_true"]))
    lines.append(rf"                \addplot[color=black, mark=none, line width=1.2pt] coordinates {{{truth_coords}}};")
    legends = ["Truth"]
    for model in models:
        if model.key not in d.columns:
            continue
        coords = " ".join(f"({_tex_num(i)},{_tex_num(y)})" for i, y in zip(d["plot_idx"], d[model.key]))
        lines.append(rf"                \addplot[color={_model_color_name(model.key)}, mark=none, line width=0.9pt] coordinates {{{coords}}};")
        legends.append(model.label)
    lines.append("                \\legend{" + ",".join(_latex_escape(x) for x in legends) + "}")
    lines.extend(
        [
            r"            \end{axis}",
            r"        \end{tikzpicture}}",
            f"    \\caption[{_latex_escape(_market_actionable_short_caption(canonical_target))}]{{{_latex_caption_escape(caption)} Selected period: {_latex_escape(date_range)}.}}",
            f"    \\label{{{label}}}",
            r"\end{figure}",
            "",
        ]
    )
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return tex_path


def _write_includegraphics_tex(*, image_path: Path, tex_path: Path, caption: str, label: str) -> Path:
    rel = image_path.as_posix()
    marker = "rq1_ml_model_benchmark/"
    if marker in rel:
        suffix = rel.split(marker, 1)[1]
        raw_prefix = "_raw_outputs/4_1_6_example_weeks/"
        if suffix.startswith(raw_prefix):
            suffix = "4_1_6_example_weeks/" + suffix[len(raw_prefix):]
        rel = "figures/4-results/rq1_ml_model_benchmark/" + suffix
    lines = [
        r"\begin{figure}[htbp]",
        r"    \centering",
        rf"    \includegraphics[width=\linewidth]{{{rel}}}",
        f"    \\caption{{{_latex_escape(caption)}}}",
        f"    \\label{{{label}}}",
        r"\end{figure}",
        "",
    ]
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return tex_path


def _prune_example_week_includegraphics_wrappers(out_dir: Path) -> list[Path]:
    removed: list[Path] = []
    for tex in sorted(out_dir.glob("**/latex_figures/**/*.tex")):
        text = tex.read_text(encoding="utf-8", errors="ignore")
        if r"\includegraphics" not in text:
            continue
        tex.unlink()
        removed.append(tex)
    return removed


def _prune_legacy_week_aliases(out_dir: Path) -> list[Path]:
    removed: list[Path] = []
    suffix_pairs = [
        ("_typical", "_typical_week"),
        ("_high_volatility", "_high_volatility_week"),
    ]
    canonical_names = {path.name for path in out_dir.glob("**/*") if path.is_file()}
    for folder_name in ["figures", "latex_figures"]:
        for path in sorted(out_dir.glob(f"**/{folder_name}/*")):
            if not path.is_file():
                continue
            stem = path.stem
            for legacy_suffix, canonical_suffix in suffix_pairs:
                if not stem.endswith(legacy_suffix):
                    continue
                canonical = path.with_name(stem[: -len(legacy_suffix)] + canonical_suffix + path.suffix)
                canonical_name = stem[: -len(legacy_suffix)] + canonical_suffix + path.suffix
                if canonical.exists() or canonical_name in canonical_names:
                    path.unlink()
                    removed.append(path)
                break
    return removed


SELECTION_METADATA_COLUMNS = [
    "weekly_std",
    "median_weekly_std",
    "abs_distance_to_median_std",
    "volatility_score",
    "selection_rank",
    "selection_scope",
    "selection_rule",
    "source",
    "week_timezone",
]


def _selection_metadata(week_meta: pd.Series | None, *, fallback_rule: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in SELECTION_METADATA_COLUMNS:
        if week_meta is not None and col in week_meta.index:
            out[col] = week_meta[col]
        elif col == "week_timezone":
            out[col] = WEEK_TIMEZONE
        elif col == "selection_rule":
            out[col] = fallback_rule
        else:
            out[col] = np.nan if col not in {"selection_scope", "source"} else ""
    return out


def build_market_actionable_examples(
    *,
    benchmark_dir: Path,
    out_dir: Path,
    models: list[ModelSpec],
    split: str,
    weeks: list[WeekSpec] | None,
    window_hours: int,
    selection_mode: str = "legacy",
    selected_weeks: pd.DataFrame | None = None,
    initial_warnings: pd.DataFrame | None = None,
    eval_origin_start: pd.Timestamp | None = None,
    eval_origin_end: pd.Timestamp | None = None,
) -> list[Path]:
    outputs: list[Path] = []
    plot_value_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    selected_weeks = _normalize_selected_week_columns(selected_weeks) if selected_weeks is not None and not selected_weeks.empty else pd.DataFrame()
    if initial_warnings is not None and not initial_warnings.empty:
        warning_rows.extend(initial_warnings.to_dict("records"))
    if selection_mode == "legacy":
        warning_rows.append(
            {
                "selection_mode": "legacy",
                "split": split,
                "target": "",
                "warning": "Legacy fixed-date example-week mode was used; weeks are parameter-selected rather than algorithmically selected.",
                "source": "manual_cli_parameters",
            }
        )

    for spec in MARKET_ACTIONABLE_SPECS:
        for canonical_target in spec["targets"]:
            target_display = _market_target_label(canonical_target)
            for model in models:
                cache[(model.key, canonical_target)] = _read_market_actionable_model_target(
                    benchmark_dir=benchmark_dir,
                    model=model,
                    split=split,
                    canonical_target=canonical_target,
                    eval_origin_start=eval_origin_start,
                    eval_origin_end=eval_origin_end,
                )
            all_model_rows = pd.concat([cache[(model.key, canonical_target)] for model in models], ignore_index=True)
            if selection_mode == "algorithmic":
                target_weeks = selected_weeks.loc[selected_weeks["target"].astype(str).eq(canonical_target)].copy()
                week_specs = [
                    WeekSpec(str(row["selection_type"]), "High-volatility" if str(row["selection_type"]) == "high_volatility_week" else "Typical", pd.Timestamp(row["week_start_utc"]))
                    for _, row in target_weeks.iterrows()
                ]
                week_meta_by_key = {str(row["selection_type"]): row for _, row in target_weeks.iterrows()}
            else:
                week_specs = list(weeks or [])
                week_meta_by_key = {}
            for week in week_specs:
                selected, diag = _select_market_actionable_rows(all_model_rows, spec=spec, week=week, window_hours=window_hours)
                week_meta = week_meta_by_key.get(week.key)
                week_start = pd.Timestamp(week_meta["week_start_utc"]) if week_meta is not None else week.start_utc
                week_end = pd.Timestamp(week_meta["week_end_utc"]) if week_meta is not None and pd.notna(week_meta["week_end_utc"]) else week.start_utc + pd.Timedelta(hours=int(window_hours))
                figure_title = _market_actionable_title(spec, _market_actionable_target_title(canonical_target))
                figure_subtitle = _market_actionable_subtitle(week, spec, canonical_target)
                caption = _market_actionable_caption(week, spec, canonical_target)
                short_caption = _market_actionable_short_caption(canonical_target)
                forecast_rule = _market_snapshot_label(spec, canonical_target, latex=False)
                selection_meta = _selection_metadata(
                    week_meta,
                    fallback_rule=str(week_meta["selection_rule"]) if week_meta is not None and "selection_rule" in week_meta.index else diag["selection_rule"],
                )
                diag_row = {
                    "selection_mode": selection_mode,
                    "selection_type": week.key,
                    "market_context": spec["market_context"],
                    "target": canonical_target,
                    "target_display": target_display,
                    "y_axis_label": target_y_axis_label(canonical_target, target_display),
                    "figure_title": figure_title,
                    "figure_subtitle": figure_subtitle,
                    "caption": caption,
                    "short_caption": short_caption,
                    "market_context_label": forecast_rule,
                    "forecast_snapshot_rule": forecast_rule,
                    "split": split,
                    "week_start_utc": week_start.isoformat(),
                    "week_end_utc": week_end.isoformat(),
                    "expected_snapshot": forecast_rule,
                    "observed_forecast_times_local": diag["observed_forecast_times_local"],
                    "observed_leads": diag["observed_leads"],
                    "n_rows": diag["n_rows"],
                    "warning": diag["warning"],
                    **selection_meta,
                }
                if diag["warning"]:
                    warning_rows.append(diag_row)
                if selected.empty:
                    continue

                tier = "result_section" if (week.key, spec["market_context"], canonical_target) in RESULT_MARKET_ACTIONABLE else "appendix"
                filename = _market_actionable_filename(spec, canonical_target, week)
                fig_path = out_dir / tier / "figures" / filename
                plot_row = _plot_market_actionable(selected=selected, spec=spec, week=week, canonical_target=canonical_target, models=models, out_path=fig_path)
                outputs.append(fig_path)
                tex_path = out_dir / tier / "latex_figures" / filename.replace(".png", ".tex")
                _write_market_actionable_latex(
                    selected=selected,
                    spec=spec,
                    week=week,
                    canonical_target=canonical_target,
                    models=models,
                    tex_path=tex_path,
                    caption=caption,
                    label=_market_actionable_figure_label(week, spec, canonical_target),
                )
                outputs.append(tex_path)
                plot_row["latex_path"] = str(tex_path)
                selected = selected.copy()
                selected["selection_mode"] = selection_mode
                selected["selection_type"] = week.key
                selected["market_context"] = spec["market_context"]
                selected["market_context_label"] = forecast_rule
                selected["figure_title"] = figure_title
                selected["figure_subtitle"] = figure_subtitle
                selected["caption"] = caption
                selected["short_caption"] = short_caption
                selected["forecast_snapshot_rule"] = forecast_rule
                selected["split"] = split
                selected["target_group"] = _target_info(_prediction_target(canonical_target))[0]
                selected["y_axis_label"] = target_y_axis_label(canonical_target, target_display)
                selected["week_start_utc"] = week_start.isoformat()
                selected["week_end_utc"] = week_end.isoformat()
                for col, value in selection_meta.items():
                    selected[col] = value
                selected["forecast_time_local"] = pd.to_datetime(selected["forecast_time_utc"], utc=True).dt.tz_convert(LOCAL_ZONE).astype(str)
                selected["target_time_local"] = pd.to_datetime(selected["target_time_utc"], utc=True).dt.tz_convert(LOCAL_ZONE).astype(str)
                selected["residual_p50"] = selected["p50"] - selected["y_true"]
                selected["abs_error_p50"] = selected["residual_p50"].abs()
                selected["figure_path"] = str(fig_path)
                if selected["p10"].isna().all() or selected["p90"].isna().all():
                    warning_rows.append({**diag_row, "warning": "missing_p10_or_p90"})
                plot_value_rows.append(
                    selected[
                        [
                            "selection_mode",
                            "selection_type",
                            "market_context",
                            "market_context_label",
                            "forecast_snapshot_rule",
                            "split",
                            "target",
                            "target_display",
                            "target_group",
                            "y_axis_label",
                            "figure_title",
                            "figure_subtitle",
                            "caption",
                            "short_caption",
                            "week_start_utc",
                            "week_end_utc",
                            *SELECTION_METADATA_COLUMNS,
                            "target_time_utc",
                            "target_time_local",
                            "forecast_time_utc",
                            "forecast_time_local",
                            "lead_time_h",
                            "model",
                            "model_label",
                            "y_true",
                            "p10",
                            "p50",
                            "p90",
                            "residual_p50",
                            "abs_error_p50",
                            "figure_path",
                        ]
                    ]
                )
                for model in models:
                    part = selected[selected["model"].eq(model.key)].copy()
                    if part.empty:
                        continue
                    metric_rows.append(
                        {
                            "selection_mode": selection_mode,
                            "selection_type": week.key,
                            "market_context": spec["market_context"],
                            "market_context_label": forecast_rule,
                            "forecast_snapshot_rule": forecast_rule,
                            "split": split,
                            "target": canonical_target,
                            "target_display": target_display,
                            "target_group": _target_info(_prediction_target(canonical_target))[0],
                            "y_axis_label": target_y_axis_label(canonical_target, target_display),
                            "figure_title": figure_title,
                            "figure_subtitle": figure_subtitle,
                            "caption": caption,
                            "short_caption": short_caption,
                            "model": model.key,
                            "model_label": model.label,
                            "mae_p50": float(part["abs_error_p50"].mean()),
                            "bias_p50": float(part["residual_p50"].mean()),
                            "median_abs_error_p50": float(part["abs_error_p50"].median()),
                            "n_obs": int(len(part)),
                            "observed_lead_min": float(part["lead_time_h"].min()),
                            "observed_lead_max": float(part["lead_time_h"].max()),
                            "week_start_utc": week_start.isoformat(),
                            "week_end_utc": week_end.isoformat(),
                            "figure_path": str(fig_path),
                            **selection_meta,
                        }
                    )
                manifest_entries.append(
                    {
                        "path": str(fig_path),
                        "artifact_type": "figure",
                        "tier": tier,
                        "target": canonical_target,
                        "target_display": target_display,
                        "y_axis_label": target_y_axis_label(canonical_target, target_display),
                        "selection_type": week.key,
                        "market_context": spec["market_context"],
                        "market_context_label": forecast_rule,
                        "figure_title": figure_title,
                        "figure_subtitle": figure_subtitle,
                        "caption": caption,
                        "short_caption": short_caption,
                        "forecast_snapshot_rule": forecast_rule,
                        "split": split,
                        "source_week_rule": str(week_meta["selection_rule"]) if week_meta is not None and "selection_rule" in week_meta.index else "legacy fixed-date week",
                        "week_timezone": selection_meta.get("week_timezone", WEEK_TIMEZONE),
                        "selection_metadata": selection_meta,
                        "thesis_use": "main thesis figure" if tier == "result_section" else "appendix figure",
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                manifest_entries.append({**manifest_entries[-1], "path": str(tex_path), "artifact_type": "latex_figure"})

    backup_csv = out_dir / "backup" / "csv"
    backup_diag = out_dir / "backup" / "diagnostics"
    backup_warn = out_dir / "backup" / "warnings"
    backup_csv.mkdir(parents=True, exist_ok=True)
    backup_diag.mkdir(parents=True, exist_ok=True)
    backup_warn.mkdir(parents=True, exist_ok=True)
    selected_weeks_path = backup_diag / "example_week_selected_weeks.csv"
    plot_values_path = backup_csv / "example_week_plot_values.csv"
    metrics_path = backup_csv / "example_week_metrics.csv"
    warn_path = backup_warn / "example_week_warnings.csv"
    selected_out = selected_weeks.copy()
    if selection_mode == "legacy":
        selected_out = pd.DataFrame(
            [
                {
                    "selection_mode": "legacy",
                    "split": split,
                    "selection_type": w.key,
                    "target": "",
                    "target_display": "",
                    "target_group": "",
                    "week_start_utc": w.start_utc.isoformat(),
                    "week_end_utc": (w.start_utc + pd.Timedelta(hours=int(window_hours))).isoformat(),
                    "weekly_std": np.nan,
                    "median_weekly_std": np.nan,
                    "abs_distance_to_median_std": np.nan,
                    "volatility_score": np.nan,
                    "n_rows": np.nan,
                    "selection_rank": 1,
                    "selection_scope": "manual",
                    "selection_rule": "Legacy fixed-date UTC week selected from CLI/default parameters.",
                    "source": "manual_cli_parameters",
                    "week_timezone": WEEK_TIMEZONE,
                }
                for w in (weeks or [])
            ]
        )
    selected_cols = [
        "selection_mode",
        "split",
        "selection_type",
        "target",
        "target_display",
        "target_group",
        "week_start_utc",
        "week_end_utc",
        "weekly_std",
        "median_weekly_std",
        "abs_distance_to_median_std",
        "volatility_score",
        "n_rows",
        "selection_rank",
        "selection_scope",
        "selection_rule",
        "source",
        "week_timezone",
    ]
    for col in selected_cols:
        if col not in selected_out.columns:
            selected_out[col] = np.nan
    selected_out[selected_cols].to_csv(selected_weeks_path, index=False)
    if plot_value_rows:
        pd.concat(plot_value_rows, ignore_index=True).to_csv(plot_values_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "selection_mode",
                "selection_type",
                "market_context",
                "market_context_label",
                "forecast_snapshot_rule",
                "split",
                "target",
                "target_display",
                "target_group",
                "y_axis_label",
                "figure_title",
                "figure_subtitle",
                "caption",
                "short_caption",
                "week_start_utc",
                "week_end_utc",
                *SELECTION_METADATA_COLUMNS,
                "target_time_utc",
                "target_time_local",
                "forecast_time_utc",
                "forecast_time_local",
                "lead_time_h",
                "model",
                "model_label",
                "y_true",
                "p10",
                "p50",
                "p90",
                "residual_p50",
                "abs_error_p50",
                "figure_path",
            ]
        ).to_csv(plot_values_path, index=False)
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(warning_rows).to_csv(warn_path, index=False)
    manifest = {
        "description": "RQ1 market-actionable example-week figures.",
        "selection_mode": selection_mode,
        "split": split,
        "benchmark_dir": str(benchmark_dir),
        "window_hours": int(window_hours),
        "week_timezone": WEEK_TIMEZONE,
        "models": [{"key": m.key, "label": m.label} for m in models],
        "artifacts": manifest_entries
        + [
            {"path": str(selected_weeks_path), "artifact_type": "diagnostics", "tier": "backup", "target": "", "selection_type": "", "market_context": "", "forecast_snapshot_rule": "", "split": split, "source_week_rule": "selected-week metadata", "thesis_use": "diagnostics", "created_at_utc": datetime.now(timezone.utc).isoformat()},
            {"path": str(plot_values_path), "artifact_type": "csv", "tier": "backup", "target": "", "selection_type": "", "market_context": "", "forecast_snapshot_rule": "", "split": split, "source_week_rule": "exact plotted values", "thesis_use": "backup data", "created_at_utc": datetime.now(timezone.utc).isoformat()},
            {"path": str(metrics_path), "artifact_type": "csv", "tier": "backup", "target": "", "selection_type": "", "market_context": "", "forecast_snapshot_rule": "", "split": split, "source_week_rule": "summary metrics", "thesis_use": "backup data", "created_at_utc": datetime.now(timezone.utc).isoformat()},
            {"path": str(warn_path), "artifact_type": "warnings", "tier": "backup", "target": "", "selection_type": "", "market_context": "", "forecast_snapshot_rule": "", "split": split, "source_week_rule": "warnings", "thesis_use": "warnings", "created_at_utc": datetime.now(timezone.utc).isoformat()},
        ],
    }
    manifest_path = out_dir / "example_week_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    outputs.extend([selected_weeks_path, plot_values_path, metrics_path, warn_path, manifest_path])
    _prune_example_week_includegraphics_wrappers(out_dir)
    _prune_legacy_week_aliases(out_dir)
    return outputs


def _target_scale_key(target: str) -> str:
    if target.endswith("_pos") or target.endswith("_neg"):
        return target.rsplit("_", 1)[0]
    return target


def _plot_value_array(view: pd.DataFrame, models: list[ModelSpec], *, target: str) -> np.ndarray:
    d = _scale_plot_frame_for_target(view, target=target, columns=["y_true", *[f"{model.key}_pred" for model in models]])
    values: list[np.ndarray] = [pd.to_numeric(d["y_true"], errors="coerce").to_numpy(dtype=float)]
    for model in models:
        col = f"{model.key}_pred"
        if col in d.columns:
            values.append(pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=float))
    if not values:
        return np.array([], dtype=float)
    arr = np.concatenate(values)
    return arr[np.isfinite(arr)]


def _padded_ylim(values: np.ndarray) -> tuple[float, float] | None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = hi - lo
    if span <= 0:
        pad = max(abs(hi) * 0.05, 1.0)
    else:
        pad = max(span * 0.06, 1e-9)
    ymin = lo - pad
    ymax = hi + pad
    if lo >= 0 and ymin < 0:
        ymin = 0.0
    return ymin, ymax


def _compute_shared_y_limits(
    views: dict[tuple[str, str], pd.DataFrame],
    *,
    targets: list[str],
    weeks: list[WeekSpec],
    models: list[ModelSpec],
) -> dict[tuple[str, str], tuple[float, float]]:
    by_scale: dict[tuple[str, str], list[np.ndarray]] = {}
    for target in targets:
        scale_key = _target_scale_key(target)
        for week in weeks:
            view = views.get((target, week.key))
            if view is None or view.empty:
                continue
            by_scale.setdefault((week.key, scale_key), []).append(_plot_value_array(view, models, target=target))

    limits: dict[tuple[str, str], tuple[float, float]] = {}
    for key, arrays in by_scale.items():
        values = np.concatenate([a for a in arrays if a.size]) if any(a.size for a in arrays) else np.array([], dtype=float)
        ylim = _padded_ylim(values)
        if ylim is not None:
            limits[key] = ylim
    return limits


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_").lower()


def _latex_escape(value: Any) -> str:
    s = str(value)
    minus_token = "@@RQ1MINUS@@"
    for label in [
        "aFRR capacity price",
        "aFRR activation price",
        "aFRR activation rate",
        "aFRR Capacity Price",
        "aFRR Activation Price",
        "aFRR Activation Rate",
        "Capacity price",
        "Activation price",
        "Activation rate",
    ]:
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


def _latex_caption_escape(value: Any) -> str:
    return _latex_escape(value).replace(r"D\$-1\$", r"D$-1$")


def _latex_color_name(role: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", role).lower()


def _tex_num(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(x):
        return "nan"
    return f"{x:.6g}"


def _plot_index_xlim(frame: pd.DataFrame) -> tuple[float, float]:
    if frame.empty:
        return 0.0, 1.0
    xmax = max(float(len(frame) - 1), 1.0)
    pad = max(1.0, min(2.0, xmax * 0.012))
    return -pad, xmax + pad


def _format_week_tick(ts: pd.Timestamp) -> tuple[str, str]:
    stamp = pd.Timestamp(ts)
    return f"{stamp.day:02d} {MONTH_ABBR[stamp.month - 1]}", f"{stamp.hour:02d}:{stamp.minute:02d}"


def _format_week_date_label(ts: pd.Timestamp) -> str:
    stamp = pd.Timestamp(ts)
    return f"{stamp.day:02d} {MONTH_ABBR[stamp.month - 1]}"


def _selected_week_local_dates(week: WeekSpec, *, n_days: int = 7) -> list[pd.Timestamp]:
    start = pd.Timestamp(week.start_utc)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    local_start = start.tz_convert(LOCAL_TZ).normalize()
    return [local_start + pd.Timedelta(days=i) for i in range(n_days)]


def _week_date_range_label(view: pd.DataFrame) -> str:
    ts = pd.to_datetime(view["target_time_utc"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return ""
    local = ts.dt.tz_convert(LOCAL_TZ)
    start = pd.Timestamp(local.min())
    end = pd.Timestamp(local.max())
    return f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}"


def _latex_week_ticks(view: pd.DataFrame, week: WeekSpec) -> tuple[str, str]:
    del week
    if view.empty:
        return "", ""
    d = view.copy()
    d["local_time"] = pd.to_datetime(d["target_time_utc"], utc=True, errors="coerce").dt.tz_convert(LOCAL_TZ)
    d = d.dropna(subset=["local_time"]).reset_index(drop=True)
    if d.empty:
        return "", ""
    d["local_date"] = d["local_time"].dt.normalize()
    first_per_date = d.groupby("local_date", sort=True)["plot_idx"].min().reset_index()
    first_per_date = first_per_date.head(7)
    xticks = [str(int(idx)) for idx in first_per_date["plot_idx"]]
    labels = [_latex_escape(_format_week_date_label(ts)) for ts in first_per_date["local_date"]]
    return ",".join(xticks), ",".join(labels)


def _model_color_name(model_key: str) -> str:
    return {"tft": "tertiary", "xgb": "primary", "linear": "secondary", "truth": "black"}.get(model_key, "neutraldark")


def write_example_week_latex(
    *,
    view: pd.DataFrame,
    target: str,
    target_label: str,
    week: WeekSpec,
    models: list[ModelSpec],
    lead_h: float,
    quantile: str,
    out_path: Path,
    ylim: tuple[float, float] | None = None,
) -> Path:
    d = view.copy()
    d["plot_idx"] = np.arange(len(d), dtype=int)
    d = _scale_plot_frame_for_target(d, target=target, columns=["y_true", *[f"{model.key}_pred" for model in models]])
    xticks, xticklabels = _latex_week_ticks(d, week)
    xmin, xmax = _plot_index_xlim(d)
    date_range = _week_date_range_label(d)
    color_lines = [
        f"\\definecolor{{{_latex_color_name(role)}}}{{HTML}}{{{hex_color.lstrip('#').upper()}}}"
        for role, hex_color in THESIS_PALETTE.items()
    ]
    y_axis_label = target_y_axis_unit_label(target)
    lines = [
        r"% Requires: \usepackage{pgfplots}",
        r"% Requires: \usepackage{xcolor}",
        r"% Recommended in preamble: \pgfplotsset{compat=1.18}",
        *color_lines,
        r"\begin{figure}[htbp]",
        r"    \centering",
        r"    \resizebox{\linewidth}{!}{%",
        r"        \begin{tikzpicture}",
        r"            \begin{axis}[",
        r"                width=0.98\textwidth,",
        r"                height=7cm,",
        rf"                title={{{_latex_escape(f'{week.label} week | {target_label} | lead={lead_h:g}h | {quantile.upper()}')}}},",
        r"                title style={at={(0.5,1.16)}, anchor=south},",
        r"                xlabel={Time},",
        rf"                ylabel={{{_latex_escape(y_axis_label)}}},",
        r"                legend style={at={(0.5,1.06)}, anchor=south, legend columns=-1, draw=none, fill=none, text=black},",
        r"                legend cell align={left},",
        r"                axis lines*=left,",
        r"                grid=major,",
        rf"                xmin={_tex_num(xmin)},",
        rf"                xmax={_tex_num(xmax)},",
        r"                enlarge x limits=false,",
        r"                clip mode=individual,",
        r"            ]",
    ]
    if xticks and xticklabels:
        insert_at = lines.index(r"                grid=major,") + 1
        lines.insert(insert_at, r"                xticklabel style={align=center, rotate=0},")
        lines.insert(insert_at, rf"                xticklabels={{{xticklabels}}},")
        lines.insert(insert_at, rf"                xtick={{{xticks}}},")
    if ylim is not None:
        insert_at = lines.index(r"                grid=major,") + 1
        lines.insert(insert_at, rf"                ymax={_tex_num(ylim[1])},")
        lines.insert(insert_at, rf"                ymin={_tex_num(ylim[0])},")
    truth_coords = " ".join(f"({_tex_num(i)},{_tex_num(y)})" for i, y in zip(d["plot_idx"], d["y_true"]))
    lines.append(rf"                \addplot[color=black, mark=none, line width=1.2pt] coordinates {{{truth_coords}}};")
    legends = ["Truth"]
    for model in models:
        col = f"{model.key}_pred"
        if col not in d.columns:
            continue
        coords = " ".join(f"({_tex_num(i)},{_tex_num(y)})" for i, y in zip(d["plot_idx"], d[col]))
        lines.append(rf"                \addplot[color={_model_color_name(model.key)}, mark=none, line width=0.9pt] coordinates {{{coords}}};")
        legends.append(model.label)
    lines.append("                \\legend{" + ",".join(_latex_escape(x) for x in legends) + "}")
    lines.extend(
        [
            r"            \end{axis}",
            r"        \end{tikzpicture}}",
            f"    \\caption{{{_latex_escape(f'{week.label} example-week forecasts for {target_label} at lead {lead_h:g}h and {quantile.upper()}. Selected period: {date_range}.')}}}",
            f"    \\label{{fig:rq1-example-week-{_safe_slug(week.key)}-{_safe_slug(target_label)}}}",
            r"\end{figure}",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def plot_example_week(
    *,
    view: pd.DataFrame,
    target: str,
    target_label: str,
    week: WeekSpec,
    models: list[ModelSpec],
    lead_h: float,
    quantile: str,
    window_hours: int,
    out_path: Path,
    ylim: tuple[float, float] | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    if view.empty:
        raise ValueError(f"Empty view for target={target}, week={week.key}.")
    d = view.copy()
    d["local_time"] = pd.to_datetime(d["target_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
    d["plot_idx"] = np.arange(len(d), dtype=int)
    d = _scale_plot_frame_for_target(d, target=target, columns=["y_true", *[f"{model.key}_pred" for model in models]])
    y_axis_label = target_y_axis_label(target, target_label)
    apply_geo_style()
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(d["plot_idx"], d["y_true"], label="Truth", color=TRUTH_COLOR, linewidth=2.4)
    row: dict[str, Any] = {
        "week": week.key,
        "week_label": week.label,
        "target": target,
        "target_label": target_label,
        "y_axis_label": y_axis_label,
        "lead_time_h": float(lead_h),
        "quantile": quantile,
        "start_utc": week.start_utc.isoformat(),
        "end_utc_exclusive": (week.start_utc + pd.Timedelta(hours=int(window_hours))).isoformat(),
        "n_rows": int(len(d)),
    }
    for model in models:
        col = f"{model.key}_pred"
        if col not in d.columns:
            continue
        mae = float(np.mean(np.abs(pd.to_numeric(d[col], errors="coerce") - pd.to_numeric(d["y_true"], errors="coerce"))))
        row[f"mae_{model.key}"] = mae
        ax.plot(
            d["plot_idx"],
            d[col],
            label=f"{model.label} {quantile.upper()} (MAE={mae:.3f})",
            color=get_model_color(model.key),
            linewidth=1.8,
        )
    title = thesis_titlecase(f"{week.label} week | {target_label} | lead={lead_h:g}h | {quantile.upper()}")
    fig.suptitle(title, y=0.965)
    ax.set_xlabel("Time")
    ax.set_ylabel(y_axis_label)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(*_plot_index_xlim(d))
    ax.margins(x=0)
    xticks, xticklabels = _latex_week_ticks(d, week)
    tick_positions = [int(x) for x in xticks.split(",") if x] if xticks else []
    tick_labels = xticklabels.split(",") if xticklabels else []
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis="x", rotation=0)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.90))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.87))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    row["figure_path"] = str(out_path)
    return row


def build_example_weeks(
    *,
    benchmark_dir: Path,
    out_dir: Path,
    models: list[ModelSpec],
    split: str,
    targets: list[str],
    weeks: list[WeekSpec],
    lead_h: float,
    quantile: str,
    window_hours: int,
    eval_origin_start: pd.Timestamp | None = None,
    eval_origin_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    rows: list[dict[str, Any]] = []
    outputs: list[Path] = []
    figures_dir = out_dir / "figures"
    ordered_targets = sorted(targets, key=target_sort_key)
    merged_by_target: dict[str, pd.DataFrame] = {}
    views: dict[tuple[str, str], pd.DataFrame] = {}
    for target in ordered_targets:
        merged_by_target[target] = load_merged_target(
            benchmark_dir=benchmark_dir,
            models=models,
            split=split,
            target=target,
            lead_h=lead_h,
            quantile=quantile,
            eval_origin_start=eval_origin_start,
            eval_origin_end=eval_origin_end,
        )
        for week in weeks:
            view = _window_slice(merged_by_target[target], week.start_utc, window_hours)
            if view.empty:
                raise ValueError(
                    f"No rows for target={target}, week={week.key}, start={week.start_utc.isoformat()}, "
                    f"lead={lead_h}, quantile={quantile}."
                )
            views[(target, week.key)] = view

    y_limits = _compute_shared_y_limits(views, targets=ordered_targets, weeks=weeks, models=models)

    for target in ordered_targets:
        target_group, target_label = _target_info(target)
        for week in weeks:
            view = views[(target, week.key)]
            ylim = y_limits.get((week.key, _target_scale_key(target)))
            out_path = figures_dir / week.key / f"{_safe_slug(target)}__lead{int(lead_h)}__{quantile}.png"
            tex_path = figures_dir / week.key / f"{_safe_slug(target)}__lead{int(lead_h)}__{quantile}.tex"
            row = plot_example_week(
                view=view,
                target=target,
                target_label=target_label,
                week=week,
                models=models,
                lead_h=lead_h,
                quantile=quantile,
                window_hours=window_hours,
                out_path=out_path,
                ylim=ylim,
            )
            write_example_week_latex(
                view=view,
                target=target,
                target_label=target_label,
                week=week,
                models=models,
                lead_h=lead_h,
                quantile=quantile,
                out_path=tex_path,
                ylim=ylim,
            )
            row["target_group"] = target_group
            row["latex_path"] = str(tex_path)
            rows.append(row)
            outputs.append(out_path)
            outputs.append(tex_path)
    summary = sort_target_frame(pd.DataFrame(rows), target_col="target", extra_cols=["week"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "example_week_metrics.csv"
    summary.to_csv(summary_path, index=False)
    outputs.append(summary_path)
    manifest_path = out_dir / "example_week_manifest.json"
    manifest = {
        "description": "RQ1 example-week truth-vs-forecast figures for all prediction targets.",
        "benchmark_dir": str(benchmark_dir),
        "split": split,
        "lead_time_h": float(lead_h),
        "quantile": quantile,
        "window_hours": int(window_hours),
        "week_timezone": WEEK_TIMEZONE,
        "models": [{"key": m.key, "label": m.label} for m in models],
        "weeks": [
            {"key": w.key, "label": w.label, "start_utc": w.start_utc.isoformat()}
            for w in weeks
        ],
        "targets": targets,
        "outputs": [str(p) for p in outputs],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    outputs.append(manifest_path)
    return summary, outputs


def _resolve_generated_path(path_value: Any, *, out_dir: Path) -> Path | None:
    if path_value is None or pd.isna(path_value):
        return None
    raw = str(path_value).strip()
    if not raw:
        return None
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(ROOT / path)
        candidates.append(out_dir / path)
    marker = "4_1_6_example_weeks/"
    if marker in raw:
        candidates.append(out_dir / raw.split(marker, 1)[1])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _append_validation_warnings(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    warn_path = out_dir / "backup" / "warnings" / "example_week_warnings.csv"
    warn_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = pd.read_csv(warn_path) if warn_path.exists() and warn_path.stat().st_size > 0 else pd.DataFrame()
    except pd.errors.EmptyDataError:
        existing = pd.DataFrame()
    pd.concat([existing, pd.DataFrame(rows)], ignore_index=True, sort=False).to_csv(warn_path, index=False)


def _includegraphics_paths(tex_path: Path) -> list[str]:
    text = tex_path.read_text(encoding="utf-8")
    return re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", text)


def validate_algorithmic_outputs(*, out_dir: Path, split: str) -> None:
    selected_path = out_dir / "backup" / "diagnostics" / "example_week_selected_weeks.csv"
    values_path = out_dir / "backup" / "csv" / "example_week_plot_values.csv"
    metrics_path = out_dir / "backup" / "csv" / "example_week_metrics.csv"
    manifest_path = out_dir / "example_week_manifest.json"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected-week metadata: {selected_path}")
    if not values_path.exists():
        raise FileNotFoundError(f"Missing exact plot values CSV: {values_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing example-week metrics CSV: {metrics_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing example-week manifest: {manifest_path}")
    selected = pd.read_csv(selected_path)
    values = pd.read_csv(values_path)
    metrics = pd.read_csv(metrics_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if selected.empty:
        raise ValueError("example_week_selected_weeks.csv is empty.")
    if values.empty:
        raise ValueError("example_week_plot_values.csv is empty.")
    if not selected["split"].astype(str).eq(split).all():
        raise ValueError("Algorithmic selected weeks contain rows outside the requested split.")
    if not values["split"].astype(str).eq(split).all():
        raise ValueError("Algorithmic plot values contain rows outside the requested split.")
    if "week_timezone" not in selected.columns or not selected["week_timezone"].astype(str).eq(WEEK_TIMEZONE).all():
        raise ValueError(f"Algorithmic selected weeks must document week_timezone={WEEK_TIMEZONE}.")
    if "week_timezone" not in values.columns or not values["week_timezone"].astype(str).eq(WEEK_TIMEZONE).all():
        raise ValueError(f"Algorithmic plot values must document week_timezone={WEEK_TIMEZONE}.")
    selected_keys = set(zip(selected["target"].astype(str), selected["selection_type"].astype(str)))
    value_keys = set(zip(values["target"].astype(str), values["selection_type"].astype(str)))
    missing = sorted(value_keys - selected_keys)
    if missing:
        raise ValueError(f"Plot values have no selected-week metadata rows for: {missing[:10]}")
    required_value_cols = set(
        SELECTION_METADATA_COLUMNS
        + [
            "y_axis_label",
            "figure_title",
            "figure_subtitle",
            "caption",
            "short_caption",
            "market_context_label",
            "forecast_snapshot_rule",
        ]
    )
    missing_value_cols = sorted(required_value_cols - set(values.columns))
    if missing_value_cols:
        raise ValueError(f"example_week_plot_values.csv is missing selection metadata columns: {missing_value_cols}")
    titles = values["figure_title"].dropna().astype(str).unique().tolist()
    invalid_titles = [title for title in titles if "|" in title or "Forecast Snapshot" in title]
    if invalid_titles:
        raise ValueError(f"Example-week figure titles still contain pipe-separated or snapshot metadata: {invalid_titles[:5]}")
    bad_title_pattern = [title for title in titles if not title.endswith(": p50 Forecast")]
    if bad_title_pattern:
        raise ValueError(f"Example-week figure titles must follow '<target display>: p50 Forecast': {bad_title_pattern[:5]}")
    captions = values["caption"].dropna().astype(str).unique().tolist()
    bad_captions: list[str] = []
    for caption in captions:
        caption_lower = caption.lower()
        if not ("typical week" in caption_lower or "high-volatility week" in caption_lower):
            bad_captions.append(caption)
            continue
        for token in ["forecast", "realized values", "p50 forecasts", "rlqr", "xgb", "tft"]:
            if token not in caption_lower:
                bad_captions.append(caption)
                break
    if bad_captions:
        raise ValueError(f"Example-week captions are missing required thesis context: {bad_captions[:5]}")
    figures = values["figure_path"].dropna().astype(str).unique().tolist()
    for fig in figures:
        part = values.loc[values["figure_path"].astype(str).eq(fig)]
        if part.empty:
            raise ValueError(f"Figure has no plot values: {fig}")
    validation_warnings: list[dict[str, Any]] = []
    paths_to_check: list[tuple[str, str]] = [("plot_values_figure_path", p) for p in figures]
    if "figure_path" in metrics.columns:
        paths_to_check.extend(("metrics_figure_path", p) for p in metrics["figure_path"].dropna().astype(str).unique().tolist())
    for artifact in manifest.get("artifacts", []):
        paths_to_check.append((f"manifest_{artifact.get('artifact_type', 'artifact')}", str(artifact.get("path", ""))))
    for source_name, path_value in paths_to_check:
        if not str(path_value).strip():
            continue
        resolved = _resolve_generated_path(path_value, out_dir=out_dir)
        if resolved is None:
            validation_warnings.append(
                {
                    "selection_mode": "algorithmic",
                    "split": split,
                    "target": "",
                    "warning": "missing_generated_artifact_path",
                    "source": source_name,
                    "path": path_value,
                }
            )
        elif resolved.suffix == ".tex":
            for include_path in _includegraphics_paths(resolved):
                if _resolve_generated_path(include_path, out_dir=out_dir) is None:
                    validation_warnings.append(
                        {
                            "selection_mode": "algorithmic",
                            "split": split,
                            "target": "",
                            "warning": "latex_wrapper_includegraphics_target_missing",
                            "source": str(resolved),
                            "path": include_path,
                        }
                    )
    if validation_warnings:
        _append_validation_warnings(out_dir, validation_warnings)
        preview = [row["path"] for row in validation_warnings[:5]]
        raise FileNotFoundError(f"Example-week validation found missing generated artifact paths: {preview}")
    da = values.loc[values["market_context"].eq("da_dminus1_11")].copy()
    if not da.empty:
        flocal = pd.to_datetime(da["forecast_time_utc"], utc=True).dt.tz_convert(LOCAL_ZONE)
        tlocal = pd.to_datetime(da["target_time_utc"], utc=True).dt.tz_convert(LOCAL_ZONE)
        if not (flocal.dt.hour.eq(11) & flocal.dt.minute.eq(0) & tlocal.dt.normalize().eq(flocal.dt.normalize() + pd.Timedelta(days=1))).all():
            raise ValueError("DA market-actionable rows do not all use D-1 11:00 Europe/Berlin.")
    bcm = values.loc[values["market_context"].eq("bcm_dplus1_08")].copy()
    if not bcm.empty:
        flocal = pd.to_datetime(bcm["forecast_time_utc"], utc=True).dt.tz_convert(LOCAL_ZONE)
        tlocal = pd.to_datetime(bcm["target_time_utc"], utc=True).dt.tz_convert(LOCAL_ZONE)
        if not (flocal.dt.hour.eq(8) & flocal.dt.minute.eq(0) & tlocal.dt.normalize().eq(flocal.dt.normalize() + pd.Timedelta(days=1))).all():
            raise ValueError("BCM market-actionable rows do not all use D-1 08:00 Europe/Berlin.")
    bem = values.loc[values["market_context"].eq("bem_h1")].copy()
    if not bem.empty and not np.isclose(pd.to_numeric(bem["lead_time_h"], errors="coerce"), 1.0, atol=1e-9).all():
        raise ValueError("BEM market-actionable rows do not all use h1.")
    for pooled in [
        {"target_afrr_capacity_price_pos", "target_afrr_capacity_price_neg"},
        {"target_afrr_activation_price_vwap_pos", "target_afrr_activation_price_vwap_neg"},
        {"target_afrr_activation_rate_pos", "target_afrr_activation_rate_neg"},
    ]:
        grouped = values.groupby(["figure_path"])["target"].agg(lambda s: set(s.astype(str)))
        bad = [fig for fig, targets in grouped.items() if pooled.issubset(targets)]
        if bad:
            raise ValueError(f"Positive and negative aFRR targets are pooled in figures: {bad[:5]}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build RQ1 example-week forecast figures.")
    p.add_argument("--benchmark-root", default="artifacts")
    p.add_argument("--benchmark-dir", default=None)
    p.add_argument("--out-dir", default="artifacts/benchmark/rq1_ml_model_benchmark/_raw_outputs/4_1_6_example_weeks")
    p.add_argument("--split", default="test", help="Main thesis/reporting split. Defaults to test.")
    p.add_argument("--models", default="tft,xgboost,linear", help="Models to compare. Defaults to all RQ1 models: TFT, XGB and RLQR.")
    p.add_argument("--targets", default="", help="Optional comma-separated prediction targets. Default: all common targets.")
    p.add_argument("--lead", type=float, default=float(DEFAULT_LEAD_H))
    p.add_argument("--quantile", default=DEFAULT_QUANTILE)
    p.add_argument("--selection-mode", choices=["algorithmic", "legacy"], default="algorithmic")
    p.add_argument("--tail-spike-selected-weeks", default="", help="Optional path to tail_spike_selected_weeks.csv for algorithmic high-volatility week reuse.")
    p.add_argument("--date", default=None, help="Optional custom UTC/local-parseable week start. If set, only this custom week is plotted.")
    p.add_argument("--typical-start", default=DEFAULT_TYPICAL_START_UTC)
    p.add_argument("--high-volatility-start", default=DEFAULT_HIGH_VOLATILITY_START_UTC)
    p.add_argument("--eval-origin-start", default=DEFAULT_EVAL_ORIGIN_START_UTC, help="Inclusive forecast-origin lower bound. Empty string disables the lower bound.")
    p.add_argument("--eval-origin-end", default=DEFAULT_EVAL_ORIGIN_END_UTC, help="Inclusive forecast-origin upper bound. Empty string disables the upper bound.")
    p.add_argument("--window-hours", type=int, default=DEFAULT_WINDOW_HOURS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    models = _parse_models(args.models)
    quantile = _parse_quantile(args.quantile)
    benchmark_dir = _discover_benchmark_dir(
        Path(args.benchmark_root),
        Path(args.benchmark_dir) if args.benchmark_dir else None,
    )
    eval_origin_start = _parse_optional_timestamp(args.eval_origin_start)
    eval_origin_end = _parse_optional_timestamp(args.eval_origin_end)
    requested_targets = [t.strip() for t in str(args.targets).split(",") if t.strip()] or None
    targets = discover_targets(benchmark_dir, split=args.split, models=models, requested_targets=requested_targets)
    out_dir = Path(args.out_dir)
    outputs: list[Path] = []
    summary = pd.DataFrame()
    if args.selection_mode == "legacy":
        weeks = build_week_specs(args)
        summary, outputs = build_example_weeks(
            benchmark_dir=benchmark_dir,
            out_dir=out_dir,
            models=models,
            split=args.split,
            targets=targets,
            weeks=weeks,
            lead_h=float(args.lead),
            quantile=quantile,
            window_hours=int(args.window_hours),
            eval_origin_start=eval_origin_start,
            eval_origin_end=eval_origin_end,
        )
        outputs.extend(
            build_market_actionable_examples(
                benchmark_dir=benchmark_dir,
                out_dir=out_dir,
                models=models,
                split=args.split,
                weeks=weeks,
                window_hours=int(args.window_hours),
                selection_mode="legacy",
                eval_origin_start=eval_origin_start,
                eval_origin_end=eval_origin_end,
            )
        )
    else:
        if args.date:
            raise ValueError("--date is legacy-only. Use --selection-mode legacy for manually parameterized weeks.")
        canonical_targets = [_canonical_target(t) for t in targets]
        selected_weeks, selection_warnings = build_algorithmic_selected_weeks(
            benchmark_dir=benchmark_dir,
            out_dir=out_dir,
            models=models,
            split=args.split,
            canonical_targets=canonical_targets,
            tail_spike_selected_weeks=str(args.tail_spike_selected_weeks).strip() or None,
            eval_origin_start=eval_origin_start,
            eval_origin_end=eval_origin_end,
        )
        legacy_warning_rows: list[dict[str, Any]] = []
        if float(args.lead) != float(DEFAULT_LEAD_H):
            legacy_warning_rows.append({"selection_mode": "algorithmic", "split": args.split, "target": "", "warning": "--lead is ignored in algorithmic mode; market-actionable snapshot rules determine the lead.", "source": "cli"})
        if quantile != DEFAULT_QUANTILE:
            legacy_warning_rows.append({"selection_mode": "algorithmic", "split": args.split, "target": "", "warning": "--quantile is ignored in algorithmic mode; final example-week plots use p50.", "source": "cli"})
        if str(args.typical_start) != DEFAULT_TYPICAL_START_UTC:
            legacy_warning_rows.append({"selection_mode": "algorithmic", "split": args.split, "target": "", "warning": "--typical-start is ignored in algorithmic mode.", "source": "cli"})
        if str(args.high_volatility_start) != DEFAULT_HIGH_VOLATILITY_START_UTC:
            legacy_warning_rows.append({"selection_mode": "algorithmic", "split": args.split, "target": "", "warning": "--high-volatility-start is ignored in algorithmic mode.", "source": "cli"})
        if legacy_warning_rows:
            selection_warnings = pd.concat([selection_warnings, pd.DataFrame(legacy_warning_rows)], ignore_index=True)
        outputs.extend(
            build_market_actionable_examples(
                benchmark_dir=benchmark_dir,
                out_dir=out_dir,
                models=models,
                split=args.split,
                weeks=None,
                window_hours=int(args.window_hours),
                selection_mode="algorithmic",
                selected_weeks=selected_weeks,
                initial_warnings=selection_warnings,
                eval_origin_start=eval_origin_start,
                eval_origin_end=eval_origin_end,
            )
        )
        validate_algorithmic_outputs(out_dir=out_dir, split=args.split)
    print("[OK] Built RQ1 example-week figures.")
    print(f"[OK] benchmark_dir={benchmark_dir}")
    print(f"[OK] selection_mode={args.selection_mode}")
    print(f"[OK] rows={len(summary)} targets={len(targets)}")
    for path in outputs:
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
