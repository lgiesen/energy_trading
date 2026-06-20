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

from energy_trading.visualization.style import apply_geo_style, get_model_color


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
    "pred_afrr_capacity_price_pos": ("aFRR capacity", "aFRR capacity price positive"),
    "target_afrr_capacity_price_pos": ("aFRR capacity", "aFRR capacity price positive"),
    "pred_afrr_capacity_price_neg": ("aFRR capacity", "aFRR capacity price negative"),
    "target_afrr_capacity_price_neg": ("aFRR capacity", "aFRR capacity price negative"),
    "pred_afrr_activation_price_pos": ("aFRR activation price", "aFRR activation price positive"),
    "target_afrr_activation_price_pos": ("aFRR activation price", "aFRR activation price positive"),
    "pred_afrr_activation_price_neg": ("aFRR activation price", "aFRR activation price negative"),
    "target_afrr_activation_price_neg": ("aFRR activation price", "aFRR activation price negative"),
    "pred_afrr_activation_rate_pos": ("aFRR activation rate", "aFRR activation rate positive"),
    "target_afrr_activation_rate_pos": ("aFRR activation rate", "aFRR activation rate positive"),
    "pred_afrr_activation_rate_neg": ("aFRR activation rate", "aFRR activation rate negative"),
    "target_afrr_activation_rate_neg": ("aFRR activation rate", "aFRR activation rate negative"),
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined_dir = benchmark_dir / "diagnostics" / "joined_predictions"
    files: dict[tuple[str, str, str], Path] = {}
    for path in sorted(joined_dir.glob("*.parquet")):
        parsed = _parse_joined_name(path)
        if parsed is not None:
            files[parsed] = path

    targets = sorted({target for _, split, target in files if split in splits})
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
                df = _read_joined_prediction(path)
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
                    }
                )

    metrics = pd.DataFrame(rows).sort_values(["split", "target_group", "target", "metric", "model_label"]).reset_index(drop=True)
    diag = pd.DataFrame(diagnostics).sort_values(["split", "target", "model"]).reset_index(drop=True)
    return metrics, diag


def _latex_escape(value: Any) -> str:
    s = str(value)
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
    return s


def _fmt_value(value: Any, *, pct: bool = False) -> str:
    try:
        x = float(value)
    except Exception:
        return "-"
    if not np.isfinite(x):
        return "-"
    return f"{x:.2f}" if pct else f"{x:.4f}"


def _fmt_blankable(value: Any, *, pct: bool = False) -> str:
    try:
        x = float(value)
    except Exception:
        return "--"
    if not np.isfinite(x):
        return "--"
    return f"{x:.2f}" if pct else f"{x:.4f}"


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
    for col in ["TFT", "XGB", "RLQR"]:
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
            for label in ["TFT", "XGB", "RLQR"]
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
        "TFT",
        "XGB",
        "RLQR",
        "best_model",
        "relative_improvement_vs_RLQR_pct",
        "n_obs",
        "quantiles_used",
        "lead_min",
        "lead_max",
    ]
    return primary[cols].sort_values(["target"]).reset_index(drop=True)


def build_detailed_table(metrics: pd.DataFrame, *, split: str) -> pd.DataFrame:
    parts = [_pivot_metric(metrics, split=split, metric=m) for m in METRICS]
    d = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
    if d.empty:
        return d
    d = _add_best_and_improvement(d).rename(columns={"n_valid_rows": "n_obs"})
    cols = [
        "target",
        "metric",
        "TFT",
        "XGB",
        "RLQR",
        "best_model",
        "relative_improvement_vs_RLQR_pct",
        "n_obs",
        "quantiles_used",
        "lead_min",
        "lead_max",
    ]
    metric_order = {m: i for i, m in enumerate(METRICS)}
    d["_metric_order"] = d["metric"].map(metric_order)
    return d[cols + ["_metric_order"]].sort_values(["target", "_metric_order"]).drop(columns=["_metric_order"]).reset_index(drop=True)


def _latex_table(
    table: pd.DataFrame,
    *,
    columns: list[str],
    headers: list[str],
    caption: str,
    label: str,
    path: Path,
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
        "        " + " & ".join(r"\textbf{" + _latex_escape(c) + "}" for c in headers) + r" \\",
        r"        \midrule",
    ]
    for _, row in d.iterrows():
        vals: list[str] = []
        for col in d.columns:
            val = row[col]
            if col in {"TFT", "XGB", "RLQR"}:
                vals.append(_fmt_value(val))
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
        columns=["target", "TFT", "XGB", "RLQR", "best_model", "relative_improvement_vs_RLQR_pct", "n_obs"],
        headers=["Target", "TFT", "XGB", "RLQR", "Best model", "Improvement vs RLQR (%)", "N"],
        caption=(
            "Full unweighted probabilistic forecast benchmark on the test split. "
            "The table reports mean pinball loss over the common quantile grid p10, p30, p50, p70 and p90. "
            "Lower values indicate better probabilistic forecast accuracy."
        ),
        label="tab:forecast_metrics_full_primary",
        path=primary_path,
    )
    _latex_table(
        detailed,
        columns=["target", "metric", "TFT", "XGB", "RLQR", "best_model", "relative_improvement_vs_RLQR_pct", "n_obs"],
        headers=["Target", "Metric", "TFT", "XGB", "RLQR", "Best model", "Improvement vs RLQR (%)", "N"],
        caption=(
            "Detailed full unweighted forecast metrics on the test split. "
            "Lower values are better for mean pinball loss, MAE and RMSE. "
            "Bias p50 is the mean p50 forecast error and indicates systematic over- or underprediction."
        ),
        label="tab:forecast_metrics_full_detailed",
        path=detailed_path,
    )
    return [primary_path, detailed_path]


def write_relative_pinball_figure(primary: pd.DataFrame, *, out_dir: Path, split: str) -> list[Path]:
    import matplotlib.pyplot as plt

    d = primary.copy()
    for col in ["TFT", "XGB", "RLQR"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d[np.isfinite(d["RLQR"]) & (d["RLQR"].abs() > 1e-12)].copy()
    if d.empty:
        return []
    apply_geo_style()
    labels = [_target_label(t) for t in d["target"]]
    x = np.arange(len(d), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6))
    for model, offset in [("TFT", -width / 2), ("XGB", width / 2)]:
        rel = d[model].to_numpy(dtype=float) / d["RLQR"].to_numpy(dtype=float)
        ax.bar(x + offset, rel, width=width, label=model, color=get_model_color(model.lower()), zorder=3)
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--", label="RLQR baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean pinball loss relative to RLQR")
    ax.set_title("Relative mean pinball loss by target (RLQR = 1; lower is better)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / f"rq1_4_1_1_forecast_metrics_full_relative_pinball_{split}.png"
    svg = fig_dir / f"rq1_4_1_1_forecast_metrics_full_relative_pinball_{split}.svg"
    pdf = fig_dir / f"rq1_4_1_1_forecast_metrics_full_relative_pinball_{split}.pdf"
    fig.savefig(png)
    fig.savefig(svg)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, svg, pdf]


def write_outputs(metrics: pd.DataFrame, diagnostics: pd.DataFrame, *, out_dir: Path, split: str) -> list[Path]:
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
    outputs.extend(write_relative_pinball_figure(primary, out_dir=out_dir, split=split))

    manifest = {
        "description": "RQ1 4.1.1 full unweighted forecast metrics. Main-text metric is mean pinball loss only.",
        "main_split": split,
        "row_intersection_key": ROW_INTERSECTION_KEY,
        "metrics": METRICS,
        "main_text_metric": "mean_pinball_loss",
        "excluded_from_4_1_1": ["winkler_score", "picp", "pinaw", "coverage", "interval_width", "lead_weighting", "gate_weighting", "tail_weighting"],
        "outputs": [str(p) for p in outputs],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    outputs.append(manifest_path)
    return outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build final RQ1 4.1.1 full unweighted forecast metrics.")
    p.add_argument("--benchmark-root", default="artifacts/forecast_benchmarks")
    p.add_argument("--benchmark-dir", action="append", default=[])
    p.add_argument("--out-dir", default="artifacts/final_benchmark/rq1/4_1_1_full_unweighted_metrics")
    p.add_argument("--split", default="test", help="Main thesis/reporting split. Defaults to test.")
    p.add_argument(
        "--splits",
        default="test",
        help="Comma-separated splits to load/export. Defaults to test only; --split selects the main reported split.",
    )
    p.add_argument("--models", default="tft,xgboost,linear", help="Models to compare. Defaults to all RQ1 models: TFT, XGB and RLQR.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    models = _parse_models(args.models)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    if args.split not in splits:
        splits.append(args.split)
    benchmark_dirs = _discover_benchmark_dirs(Path(args.benchmark_root), [Path(p) for p in args.benchmark_dir])
    benchmark_dir = benchmark_dirs[0]
    metrics, diagnostics = build_full_metrics(benchmark_dir=benchmark_dir, models=models, splits=splits)
    outputs = write_outputs(metrics, diagnostics, out_dir=Path(args.out_dir), split=args.split)
    print("[OK] Built RQ1 4.1.1 full unweighted forecast metrics.")
    print(f"[OK] benchmark_dir={benchmark_dir}")
    for path in outputs:
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
