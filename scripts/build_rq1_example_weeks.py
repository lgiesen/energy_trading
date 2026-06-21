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

from energy_trading.visualization.style import THESIS_PALETTE, apply_geo_style, get_model_color, thesis_titlecase
from energy_trading.evaluation.rq1_target_order import model_sort_key, sort_target_frame, target_sort_key


DEFAULT_HIGH_VOLATILITY_START_UTC = "2025-10-05T22:00:00Z"
DEFAULT_TYPICAL_START_UTC = "2025-03-30T22:00:00Z"
DEFAULT_LEAD_H = 24
DEFAULT_QUANTILE = "p50"
DEFAULT_WINDOW_HOURS = 24 * 7
LOCAL_TZ = "Europe/Berlin"
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


def _target_info(target: str) -> tuple[str, str]:
    return TARGET_LABELS.get(target, ("Other", target.replace("_", " ")))


def _window_slice(df: pd.DataFrame, start_utc: pd.Timestamp, window_hours: int) -> pd.DataFrame:
    end_utc = start_utc + pd.Timedelta(hours=int(window_hours))
    ts = pd.to_datetime(df["target_time_utc"], utc=True, errors="coerce")
    return df.loc[(ts >= start_utc) & (ts < end_utc)].copy()


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
    p.add_argument("--out-dir", default="artifacts/rq1_ml_model_benchmark/_raw_outputs/4_1_6_example_weeks")
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
