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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_trading.visualization.style import apply_geo_style, get_model_color


DEFAULT_HIGH_VOLATILITY_START_UTC = "2025-10-05T22:00:00Z"
DEFAULT_TYPICAL_START_UTC = "2025-03-30T22:00:00Z"
DEFAULT_LEAD_H = 24
DEFAULT_QUANTILE = "p50"
DEFAULT_WINDOW_HOURS = 24 * 7
LOCAL_TZ = "Europe/Berlin"

MODEL_KEYS = {
    "tft": ("tft", "TFT"),
    "xgb": ("xgb", "XGB"),
    "xgboost": ("xgb", "XGB"),
    "linear": ("linear", "RLQR"),
    "rlqr": ("linear", "RLQR"),
}

TARGET_LABELS = {
    "pred_da_price": ("DA", "DA price"),
    "pred_afrr_capacity_price_pos": ("aFRR capacity", "aFRR capacity price positive"),
    "pred_afrr_capacity_price_neg": ("aFRR capacity", "aFRR capacity price negative"),
    "pred_afrr_activation_price_pos": ("aFRR activation price", "aFRR activation price positive"),
    "pred_afrr_activation_price_neg": ("aFRR activation price", "aFRR activation price negative"),
    "pred_afrr_activation_rate_pos": ("aFRR activation rate", "aFRR activation rate positive"),
    "pred_afrr_activation_rate_neg": ("aFRR activation rate", "aFRR activation rate negative"),
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
    return out


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


def _target_info(target: str) -> tuple[str, str]:
    return TARGET_LABELS.get(target, ("Other", target.replace("_", " ")))


def _window_slice(df: pd.DataFrame, start_utc: pd.Timestamp, window_hours: int) -> pd.DataFrame:
    end_utc = start_utc + pd.Timedelta(hours=int(window_hours))
    ts = pd.to_datetime(df["target_time_utc"], utc=True, errors="coerce")
    return df.loc[(ts >= start_utc) & (ts < end_utc)].copy()


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_").lower()


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
    ax.set_title(f"{week.label} week | {target_label} | lead={lead_h:g}h | {quantile.upper()}")
    ax.set_xlabel(f"Local time ({LOCAL_TZ})")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=20)
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
    for target in targets:
        target_group, target_label = _target_info(target)
        merged = load_merged_target(
            benchmark_dir=benchmark_dir,
            models=models,
            split=split,
            target=target,
            lead_h=lead_h,
            quantile=quantile,
        )
        for week in weeks:
            view = _window_slice(merged, week.start_utc, window_hours)
            if view.empty:
                raise ValueError(
                    f"No rows for target={target}, week={week.key}, start={week.start_utc.isoformat()}, "
                    f"lead={lead_h}, quantile={quantile}."
                )
            out_path = figures_dir / week.key / f"{_safe_slug(target)}__lead{int(lead_h)}__{quantile}.png"
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
            )
            row["target_group"] = target_group
            rows.append(row)
            outputs.append(out_path)
    summary = pd.DataFrame(rows).sort_values(["week", "target_group", "target"]).reset_index(drop=True)
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
    p.add_argument("--benchmark-root", default="artifacts/forecast_benchmarks")
    p.add_argument("--benchmark-dir", default=None)
    p.add_argument("--out-dir", default="artifacts/final_benchmark/_raw_outputs/4_1_7_example_weeks")
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
    print("[OK] Built RQ1 example-week figures.")
    print(f"[OK] benchmark_dir={benchmark_dir}")
    print(f"[OK] rows={len(summary)} targets={len(targets)} weeks={len(weeks)}")
    for path in outputs:
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
