#!/usr/bin/env python3
"""Build RQ1 example-week truth-vs-forecast figures for all targets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
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
DEFAULT_LEAD_H = 24
DEFAULT_QUANTILE = "p50"
DEFAULT_WINDOW_HOURS = 24 * 7
LOCAL_TZ = "Europe/Berlin"
LOCAL_ZONE = ZoneInfo(LOCAL_TZ)
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

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
    ("typical", "da_dminus1_11", "target_da_price"),
    ("typical", "bcm_dplus1_08", "target_afrr_capacity_price_pos"),
    ("typical", "bcm_dplus1_08", "target_afrr_capacity_price_neg"),
    ("typical", "bcm_dplus1_08", "target_afrr_activation_price_vwap_pos"),
    ("typical", "bcm_dplus1_08", "target_afrr_activation_price_vwap_neg"),
    ("typical", "bem_h1", "target_afrr_activation_price_vwap_pos"),
    ("typical", "bem_h1", "target_afrr_activation_price_vwap_neg"),
    ("typical", "bem_h1", "target_afrr_activation_rate_pos"),
    ("typical", "bem_h1", "target_afrr_activation_rate_neg"),
    ("high_volatility", "da_dminus1_11", "target_da_price"),
    ("high_volatility", "bcm_dplus1_08", "target_afrr_capacity_price_pos"),
    ("high_volatility", "bcm_dplus1_08", "target_afrr_capacity_price_neg"),
    ("high_volatility", "bcm_dplus1_08", "target_afrr_activation_price_vwap_pos"),
    ("high_volatility", "bcm_dplus1_08", "target_afrr_activation_price_vwap_neg"),
    ("high_volatility", "bem_h1", "target_afrr_activation_price_vwap_pos"),
    ("high_volatility", "bem_h1", "target_afrr_activation_price_vwap_neg"),
    ("high_volatility", "bem_h1", "target_afrr_activation_rate_pos"),
    ("high_volatility", "bem_h1", "target_afrr_activation_rate_neg"),
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
) -> pd.DataFrame:
    path = benchmark_dir / "diagnostics" / "joined_predictions" / f"{model.key}__{split}__{target}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing joined prediction file: {path}")
    df = pd.read_parquet(path)
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
    out = df[["target_time_utc", "lead_time_h", "y_true", pred_col]].copy()
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


def _window_slice(df: pd.DataFrame, start_utc: pd.Timestamp, window_hours: int) -> pd.DataFrame:
    end_utc = start_utc + pd.Timedelta(hours=int(window_hours))
    ts = pd.to_datetime(df["target_time_utc"], utc=True, errors="coerce")
    return df.loc[(ts >= start_utc) & (ts < end_utc)].copy()


def _read_market_actionable_model_target(
    *,
    benchmark_dir: Path,
    model: ModelSpec,
    split: str,
    canonical_target: str,
) -> pd.DataFrame:
    target = _prediction_target(canonical_target)
    path = benchmark_dir / "diagnostics" / "joined_predictions" / f"{model.key}__{split}__{target}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing joined prediction file: {path}")
    df = pd.read_parquet(path)
    if "forecast_time_utc" not in df.columns and "snapshot_time_utc" in df.columns:
        df = df.rename(columns={"snapshot_time_utc": "forecast_time_utc"})
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
    out = df[["forecast_time_utc", "target_time_utc", "lead_time_h", "y_true", pred_col]].copy()
    out["forecast_time_utc"] = pd.to_datetime(out["forecast_time_utc"], utc=True, errors="coerce")
    out["target_time_utc"] = pd.to_datetime(out["target_time_utc"], utc=True, errors="coerce")
    out["lead_time_h"] = pd.to_numeric(out["lead_time_h"], errors="coerce")
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out["p50"] = pd.to_numeric(out[pred_col], errors="coerce")
    out = out.dropna(subset=["forecast_time_utc", "target_time_utc", "lead_time_h", "y_true", "p50"]).copy()
    out["model"] = model.key
    out["model_label"] = model.label
    out["target"] = canonical_target
    out["target_display"] = _market_target_label(canonical_target)
    return out[["model", "model_label", "target", "target_display", "forecast_time_utc", "target_time_utc", "lead_time_h", "y_true", "p50"]]


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
    if spec["market_context"] == "da_dminus1_11":
        return "DA price | D-1 11:00 forecast snapshot | p50"
    return f"{spec['title_prefix']} | {target_display} | p50"


def _market_actionable_caption(week: WeekSpec, spec: dict[str, Any], target_display: str) -> str:
    return (
        f"Example-week {target_display} forecasts for the selected {week.label.lower()} week using the "
        f"{spec['snapshot_description']} The figure compares realized values with p50 forecasts from RLQR, XGB and TFT and therefore shows information available at the market decision time."
    )


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
    target_display = _market_target_label(canonical_target)
    apply_geo_style()
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    ax.plot(d["local_time"], d["y_true"], label="Truth", color=get_model_color("truth"), linewidth=2.3)
    row: dict[str, Any] = {
        "week_type": week.key,
        "market_context": spec["market_context"],
        "target": canonical_target,
        "target_display": target_display,
        "figure_path": str(out_path),
    }
    for model in models:
        if model.key not in d.columns:
            continue
        mae = float(np.mean(np.abs(pd.to_numeric(d[model.key], errors="coerce") - pd.to_numeric(d["y_true"], errors="coerce"))))
        ax.plot(d["local_time"], d[model.key], label=f"{model.label} p50 (MAE={mae:.3f})", color=get_model_color(model.key), linewidth=1.8)
        row[f"mae_{model.key}"] = mae
    ax.set_title(thesis_titlecase(_market_actionable_title(spec, target_display)))
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    tick_rows = d.index[d["local_time"].dt.hour.eq(0) & d["local_time"].dt.minute.eq(0)].tolist()
    if not tick_rows:
        tick_rows = list(range(0, len(d), max(1, len(d) // 7)))
    tick_positions = [d.loc[i, "local_time"] for i in tick_rows if i in d.index]
    tick_labels = ["\n".join(_format_week_tick(pd.Timestamp(t))) for t in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.24))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return row


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


def build_market_actionable_examples(
    *,
    benchmark_dir: Path,
    out_dir: Path,
    models: list[ModelSpec],
    split: str,
    weeks: list[WeekSpec],
    window_hours: int,
) -> list[Path]:
    outputs: list[Path] = []
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, str], pd.DataFrame] = {}

    for spec in MARKET_ACTIONABLE_SPECS:
        for canonical_target in spec["targets"]:
            target_display = _market_target_label(canonical_target)
            for model in models:
                cache[(model.key, canonical_target)] = _read_market_actionable_model_target(
                    benchmark_dir=benchmark_dir,
                    model=model,
                    split=split,
                    canonical_target=canonical_target,
                )
            all_model_rows = pd.concat([cache[(model.key, canonical_target)] for model in models], ignore_index=True)
            for week in weeks:
                selected, diag = _select_market_actionable_rows(all_model_rows, spec=spec, week=week, window_hours=window_hours)
                diag_row = {
                    "week_type": week.key,
                    "market_context": spec["market_context"],
                    "target": canonical_target,
                    "selection_rule": diag["selection_rule"],
                    "expected_snapshot": diag["expected_snapshot"],
                    "observed_forecast_times_local": diag["observed_forecast_times_local"],
                    "observed_leads": diag["observed_leads"],
                    "n_rows": diag["n_rows"],
                    "warning": diag["warning"],
                }
                diagnostic_rows.append(diag_row)
                if diag["warning"]:
                    warning_rows.append(diag_row)
                if selected.empty:
                    continue
                selected = selected.copy()
                selected["week_type"] = week.key
                selected["market_context"] = spec["market_context"]
                selected["forecast_time_local"] = pd.to_datetime(selected["forecast_time_utc"], utc=True).dt.tz_convert(LOCAL_ZONE).astype(str)
                selected["target_time_local"] = pd.to_datetime(selected["target_time_utc"], utc=True).dt.tz_convert(LOCAL_ZONE).astype(str)
                selected["error_p50"] = selected["p50"] - selected["y_true"]
                selected["abs_error_p50"] = selected["error_p50"].abs()
                prediction_rows.append(
                    selected[
                        [
                            "week_type",
                            "market_context",
                            "target",
                            "target_display",
                            "model_label",
                            "forecast_time_utc",
                            "forecast_time_local",
                            "target_time_utc",
                            "target_time_local",
                            "lead_time_h",
                            "y_true",
                            "p50",
                            "error_p50",
                            "abs_error_p50",
                        ]
                    ].rename(columns={"model_label": "model"})
                )
                for model in models:
                    part = selected[selected["model"].eq(model.key)].copy()
                    if part.empty:
                        continue
                    metric_rows.append(
                        {
                            "week_type": week.key,
                            "market_context": spec["market_context"],
                            "target": canonical_target,
                            "target_display": target_display,
                            "model": model.label,
                            "mae_p50": float(part["abs_error_p50"].mean()),
                            "bias_p50": float(part["error_p50"].mean()),
                            "median_abs_error_p50": float(part["abs_error_p50"].median()),
                            "n_obs": int(len(part)),
                            "observed_lead_min": float(part["lead_time_h"].min()),
                            "observed_lead_max": float(part["lead_time_h"].max()),
                            "forecast_snapshot_description": spec["snapshot_description"],
                        }
                    )

                tier = "result_section" if (week.key, spec["market_context"], canonical_target) in RESULT_MARKET_ACTIONABLE else "appendix"
                filename = _market_actionable_filename(spec, canonical_target, week)
                fig_path = out_dir / tier / "figures" / filename
                plot_row = _plot_market_actionable(selected=selected, spec=spec, week=week, canonical_target=canonical_target, models=models, out_path=fig_path)
                outputs.append(fig_path)
                tex_path = out_dir / tier / "latex_figures" / filename.replace(".png", ".tex")
                _write_includegraphics_tex(
                    image_path=fig_path,
                    tex_path=tex_path,
                    caption=_market_actionable_caption(week, spec, target_display),
                    label=f"fig:rq1-market-actionable-{_safe_slug(week.key)}-{_safe_slug(spec['market_context'])}-{_safe_slug(target_display)}",
                )
                outputs.append(tex_path)
                plot_row["latex_path"] = str(tex_path)

    backup_csv = out_dir / "backup" / "csv"
    backup_diag = out_dir / "backup" / "diagnostics"
    backup_warn = out_dir / "backup" / "warnings"
    backup_csv.mkdir(parents=True, exist_ok=True)
    backup_diag.mkdir(parents=True, exist_ok=True)
    backup_warn.mkdir(parents=True, exist_ok=True)
    pred_path = backup_csv / "example_week_market_actionable_predictions.csv"
    metrics_path = backup_csv / "example_week_market_actionable_metrics.csv"
    diag_path = backup_diag / "example_week_market_actionable_selection.csv"
    warn_path = backup_warn / "example_week_market_actionable_warnings.csv"
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(pred_path, index=False)
    else:
        pd.DataFrame(columns=["week_type", "market_context", "target", "target_display", "model", "forecast_time_utc", "forecast_time_local", "target_time_utc", "target_time_local", "lead_time_h", "y_true", "p50", "error_p50", "abs_error_p50"]).to_csv(pred_path, index=False)
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(diagnostic_rows).to_csv(diag_path, index=False)
    pd.DataFrame(warning_rows).to_csv(warn_path, index=False)
    outputs.extend([pred_path, metrics_path, diag_path, warn_path])
    return outputs


def _target_scale_key(target: str) -> str:
    if target.endswith("_pos") or target.endswith("_neg"):
        return target.rsplit("_", 1)[0]
    return target


def _plot_value_array(view: pd.DataFrame, models: list[ModelSpec]) -> np.ndarray:
    values: list[np.ndarray] = [pd.to_numeric(view["y_true"], errors="coerce").to_numpy(dtype=float)]
    for model in models:
        col = f"{model.key}_pred"
        if col in view.columns:
            values.append(pd.to_numeric(view[col], errors="coerce").to_numpy(dtype=float))
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
            by_scale.setdefault((week.key, scale_key), []).append(_plot_value_array(view, models))

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


def _format_week_tick(ts: pd.Timestamp) -> tuple[str, str]:
    stamp = pd.Timestamp(ts)
    return f"{stamp.day:02d} {MONTH_ABBR[stamp.month - 1]}", f"{stamp.hour:02d}:{stamp.minute:02d}"


def _week_date_range_label(view: pd.DataFrame) -> str:
    ts = pd.to_datetime(view["target_time_utc"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return ""
    local = ts.dt.tz_convert(LOCAL_TZ)
    start = pd.Timestamp(local.min())
    end = pd.Timestamp(local.max())
    return f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}"


def _latex_week_ticks(view: pd.DataFrame) -> tuple[str, str]:
    if view.empty:
        return "", ""
    d = view.copy()
    d["local_time"] = pd.to_datetime(d["target_time_utc"], utc=True, errors="coerce").dt.tz_convert(LOCAL_TZ)
    d = d.dropna(subset=["local_time"]).reset_index(drop=True)
    if d.empty:
        return "", ""
    tick_rows = d.index[d["local_time"].dt.hour.eq(0) & d["local_time"].dt.minute.eq(0)].tolist()
    if not tick_rows:
        tick_rows = list(range(0, len(d), 24))
    xticks: list[str] = []
    labels: list[str] = []
    for idx in tick_rows:
        if idx < 0 or idx >= len(d):
            continue
        date_part, time_part = _format_week_tick(pd.Timestamp(d.loc[idx, "local_time"]))
        xticks.append(str(int(idx)))
        labels.append(rf"\shortstack{{{date_part}\\{time_part}}}")
    return ",".join(xticks), ",".join(labels)


def _model_color_name(model_key: str) -> str:
    return {"tft": "tertiary", "xgb": "primary", "linear": "secondary", "truth": "perfectforesight"}.get(model_key, "neutraldark")


def write_example_week_latex(
    *,
    view: pd.DataFrame,
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
    xticks, xticklabels = _latex_week_ticks(d)
    date_range = _week_date_range_label(d)
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
        rf"                title={{{_latex_escape(f'{week.label} week | {target_label} | lead={lead_h:g}h | {quantile.upper()}')}}},",
        r"                xlabel={Time},",
        r"                ylabel={Value},",
        r"                legend style={at={(0.5,1.10)}, anchor=south, legend columns=-1, draw=none, fill=none, text=black},",
        r"                legend cell align={left},",
        r"                axis lines*=left,",
        r"                grid=major,",
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
    lines.append(rf"                \addplot[color=perfectforesight, mark=none, line width=1.2pt] coordinates {{{truth_coords}}};")
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
    apply_geo_style()
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(d["local_time"], d["y_true"], label="Truth", color=get_model_color("truth"), linewidth=2.4)
    row: dict[str, Any] = {
        "week": week.key,
        "week_label": week.label,
        "target": target,
        "target_label": target_label,
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
            d["local_time"],
            d[col],
            label=f"{model.label} {quantile.upper()} (MAE={mae:.3f})",
            color=get_model_color(model.key),
            linewidth=1.8,
        )
    ax.set_title(thesis_titlecase(f"{week.label} week | {target_label} | lead={lead_h:g}h | {quantile.upper()}"))
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    if ylim is not None:
        ax.set_ylim(*ylim)
    tick_rows = d.index[d["local_time"].dt.hour.eq(0) & d["local_time"].dt.minute.eq(0)].tolist()
    if not tick_rows:
        tick_rows = list(range(0, len(d), 24))
    tick_positions = [d.loc[i, "local_time"] for i in tick_rows if i in d.index]
    tick_labels = ["\n".join(_format_week_tick(pd.Timestamp(t))) for t in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.24))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
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
    p.add_argument("--date", default=None, help="Optional custom UTC/local-parseable week start. If set, only this custom week is plotted.")
    p.add_argument("--typical-start", default=DEFAULT_TYPICAL_START_UTC)
    p.add_argument("--high-volatility-start", default=DEFAULT_HIGH_VOLATILITY_START_UTC)
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
    requested_targets = [t.strip() for t in str(args.targets).split(",") if t.strip()] or None
    targets = discover_targets(benchmark_dir, split=args.split, models=models, requested_targets=requested_targets)
    weeks = build_week_specs(args)
    summary, outputs = build_example_weeks(
        benchmark_dir=benchmark_dir,
        out_dir=Path(args.out_dir),
        models=models,
        split=args.split,
        targets=targets,
        weeks=weeks,
        lead_h=float(args.lead),
        quantile=quantile,
        window_hours=int(args.window_hours),
    )
    market_outputs = build_market_actionable_examples(
        benchmark_dir=benchmark_dir,
        out_dir=Path(args.out_dir),
        models=models,
        split=args.split,
        weeks=weeks,
        window_hours=int(args.window_hours),
    )
    outputs.extend(market_outputs)
    print("[OK] Built RQ1 example-week figures.")
    print(f"[OK] benchmark_dir={benchmark_dir}")
    print(f"[OK] rows={len(summary)} targets={len(targets)} weeks={len(weeks)}")
    for path in outputs:
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
