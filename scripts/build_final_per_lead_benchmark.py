#!/usr/bin/env python3
"""Build RQ1 4.1.3 per-lead-hour benchmark outputs.

This script reads existing joined forecast benchmark predictions, aligns TFT,
XGB and RLQR on a common valid row intersection, and reports unweighted
per-lead forecast metrics. It does not train models, run simulations, or use
lead/gate/tail/sample weights.
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_trading.visualization.style import apply_geo_style, get_model_color, thesis_titlecase
from energy_trading.evaluation.rq1_target_order import MODEL_LABELS, model_sort_key, ordered_unique, sort_target_frame, target_sort_key


MODEL_KEYS = {
    "tft": ("tft", "TFT"),
    "xgb": ("xgb", "XGB"),
    "xgboost": ("xgb", "XGB"),
    "linear": ("linear", "RLQR"),
    "rlqr": ("linear", "RLQR"),
}

TARGET_GROUPS = {
    "pred_da_price": ("da_price", "DA price", "DA price"),
    "target_da_price": ("da_price", "DA price", "DA price"),
    "pred_afrr_capacity_price_pos": ("afrr_capacity_price", "aFRR capacity price", "aFRR capacity price +"),
    "target_afrr_capacity_price_pos": ("afrr_capacity_price", "aFRR capacity price", "aFRR capacity price +"),
    "pred_afrr_capacity_price_neg": ("afrr_capacity_price", "aFRR capacity price", "aFRR capacity price -"),
    "target_afrr_capacity_price_neg": ("afrr_capacity_price", "aFRR capacity price", "aFRR capacity price -"),
    "pred_afrr_activation_price_pos": ("afrr_activation_price", "aFRR activation price", "aFRR activation price +"),
    "target_afrr_activation_price_pos": ("afrr_activation_price", "aFRR activation price", "aFRR activation price +"),
    "target_afrr_activation_price_vwap_pos": ("afrr_activation_price", "aFRR activation price", "aFRR activation price +"),
    "pred_afrr_activation_price_neg": ("afrr_activation_price", "aFRR activation price", "aFRR activation price -"),
    "target_afrr_activation_price_neg": ("afrr_activation_price", "aFRR activation price", "aFRR activation price -"),
    "target_afrr_activation_price_vwap_neg": ("afrr_activation_price", "aFRR activation price", "aFRR activation price -"),
    "pred_afrr_activation_rate_pos": ("afrr_activation_rate", "aFRR activation rate", "aFRR activation rate +"),
    "target_afrr_activation_rate_pos": ("afrr_activation_rate", "aFRR activation rate", "aFRR activation rate +"),
    "pred_afrr_activation_rate_neg": ("afrr_activation_rate", "aFRR activation rate", "aFRR activation rate -"),
    "target_afrr_activation_rate_neg": ("afrr_activation_rate", "aFRR activation rate", "aFRR activation rate -"),
}

GROUP_ORDER = ["da_price", "afrr_capacity_price", "afrr_activation_price", "afrr_activation_rate"]
GROUP_FILE_STEMS = {
    "da_price": "da_price",
    "afrr_capacity_price": "afrr_capacity_price",
    "afrr_activation_price": "afrr_activation_price",
    "afrr_activation_rate": "afrr_activation_rate",
}
METRIC_LABELS = {
    "mean_pinball_loss": "Mean pinball loss",
    "mae_p50": "MAE p50",
    "rmse_p50": "RMSE p50",
    "bias_p50": "Bias p50",
    "coverage_p10_p90": "p10-p90 empirical coverage",
    "coverage_p30_p70": "p30-p70 empirical coverage",
}
MAIN_METRICS = ["mean_pinball_loss", "mae_p50", "rmse_p50"]
LEAD_RANGES = [
    ("short_h1_8", 1, 8),
    ("medium_h9_16", 9, 16),
    ("long_h17_48", 17, 48),
]
LEAD_RANGE_LABELS = {
    "short_h1_8": "h1-h8",
    "medium_h9_16": "h9-h16",
    "long_h17_48": "h17-h48",
}
QCOL_RE = re.compile(r"^p(\d{1,2})$", re.IGNORECASE)
DEFAULT_EVAL_ORIGIN_START_UTC = "2025-01-13T23:00:00Z"
DEFAULT_EVAL_ORIGIN_END_UTC = "2026-02-26T21:00:00Z"


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
        if canonical not in seen:
            out.append(ModelSpec(canonical, label))
            seen.add(canonical)
    if not out:
        raise ValueError("At least one model is required.")
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
        raise FileNotFoundError(
            f"Missing joined predictions directory: {joined}. "
            "Run scripts/run_forecast_benchmark.py with --save-joined-predictions first."
        )
    return out


def _parse_joined_name(path: Path) -> tuple[str, str, str] | None:
    parts = path.stem.split("__", 2)
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


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


def _target_info(target: str) -> tuple[str, str, str]:
    return TARGET_GROUPS.get(str(target), ("other", "Other", str(target).replace("_", " ")))


def _read_joined(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    required = {"target_time_utc", "lead_time_h", "y_true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df["target_time_utc"] = pd.to_datetime(df["target_time_utc"], utc=True, errors="coerce")
    if "forecast_time_utc" in df.columns:
        df["forecast_time_utc"] = pd.to_datetime(df["forecast_time_utc"], utc=True, errors="coerce")
    elif "snapshot_time_utc" in df.columns:
        df["forecast_time_utc"] = pd.to_datetime(df["snapshot_time_utc"], utc=True, errors="coerce")
    else:
        df["forecast_time_utc"] = df["target_time_utc"] - pd.to_timedelta(df["lead_time_h"], unit="h")
    df["lead_time_h"] = pd.to_numeric(df["lead_time_h"], errors="coerce")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    if "p50" not in df.columns and "predicted_value" in df.columns:
        df["p50"] = pd.to_numeric(df["predicted_value"], errors="coerce")
    if "p50" not in df.columns:
        raise ValueError(f"{path} must contain p50 or predicted_value.")
    return df


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


def _key_cols(loaded: dict[str, pd.DataFrame]) -> list[str]:
    has_forecast_time = all(
        "forecast_time_utc" in df.columns and df["forecast_time_utc"].notna().any()
        for df in loaded.values()
    )
    return ["forecast_time_utc", "target_time_utc", "lead_time_h"] if has_forecast_time else ["target_time_utc", "lead_time_h"]


def _valid_frame(df: pd.DataFrame, qcols: dict[float, str], key_cols: list[str]) -> pd.DataFrame:
    cols = list(dict.fromkeys([*key_cols, "y_true", "p50", *[qcols[q] for q in sorted(qcols)]]))
    out = df[cols].copy()
    for col in ["y_true", "p50", *[qcols[q] for q in sorted(qcols)]]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    mask = pd.Series(True, index=out.index)
    for col in key_cols:
        mask &= out[col].notna()
    for col in ["y_true", "p50", *[qcols[q] for q in sorted(qcols)]]:
        mask &= np.isfinite(out[col].to_numpy(dtype=float))
    return out.loc[mask].copy()


def _key_tuples(df: pd.DataFrame, key_cols: list[str]) -> set[tuple[Any, ...]]:
    return set(map(tuple, df[key_cols].itertuples(index=False, name=None)))


def _pinball_values(y: np.ndarray, pred: np.ndarray, q: float) -> np.ndarray:
    err = y - pred
    return np.maximum(q * err, (q - 1.0) * err)


def _metrics_for_frame(df: pd.DataFrame, qcols: dict[float, str]) -> dict[str, float]:
    y = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    p50 = pd.to_numeric(df["p50"], errors="coerce").to_numpy(dtype=float)
    pinball_parts = [
        _pinball_values(y, pd.to_numeric(df[qcols[q]], errors="coerce").to_numpy(dtype=float), q)
        for q in sorted(qcols)
    ]
    out = {
        "mean_pinball_loss": float(np.mean(np.vstack(pinball_parts))) if pinball_parts else float("nan"),
        "mae_p50": float(np.mean(np.abs(p50 - y))),
        "rmse_p50": float(np.sqrt(np.mean((p50 - y) ** 2))),
        "bias_p50": float(np.mean(p50 - y)),
    }
    if 0.10 in qcols and 0.90 in qcols:
        lo = pd.to_numeric(df[qcols[0.10]], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(df[qcols[0.90]], errors="coerce").to_numpy(dtype=float)
        out["coverage_p10_p90"] = float(np.mean((lo <= y) & (y <= hi)))
    if 0.30 in qcols and 0.70 in qcols:
        lo = pd.to_numeric(df[qcols[0.30]], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(df[qcols[0.70]], errors="coerce").to_numpy(dtype=float)
        out["coverage_p30_p70"] = float(np.mean((lo <= y) & (y <= hi)))
    return out


def assign_lead_range(lead_time_h: float) -> str | None:
    try:
        lead = int(float(lead_time_h))
    except Exception:
        return None
    for name, lo, hi in LEAD_RANGES:
        if lo <= lead <= hi:
            return name
    return None


def build_per_lead_outputs(
    *,
    benchmark_dir: Path,
    models: list[ModelSpec],
    splits: list[str],
    horizon: int,
    eval_origin_start: pd.Timestamp | None,
    eval_origin_end: pd.Timestamp | None,
) -> dict[str, pd.DataFrame]:
    joined_dir = benchmark_dir / "diagnostics" / "joined_predictions"
    files: dict[tuple[str, str, str], Path] = {}
    for path in sorted(joined_dir.glob("*.parquet")):
        parsed = _parse_joined_name(path)
        if parsed is not None:
            files[parsed] = path

    targets = sorted({target for _, split, target in files if split in set(splits)}, key=target_sort_key)
    if not targets:
        raise FileNotFoundError(f"No joined prediction parquet files for splits={splits} in {joined_dir}.")

    metric_rows: list[dict[str, Any]] = []
    row_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    expected_leads = set(range(1, int(horizon) + 1))

    for split in splits:
        for target in targets:
            loaded: dict[str, pd.DataFrame] = {}
            qmaps: dict[str, dict[float, str]] = {}
            for model in models:
                path = files.get((model.key, split, target))
                if path is None:
                    raise FileNotFoundError(f"Missing joined predictions for model={model.key}, split={split}, target={target}.")
                df = _apply_forecast_origin_window(
                    _read_joined(path),
                    start=eval_origin_start,
                    end=eval_origin_end,
                )
                loaded[model.key] = df
                qmaps[model.key] = _quantile_cols(df)
                if 0.50 not in qmaps[model.key]:
                    raise ValueError(f"Missing p50 for model={model.key}, split={split}, target={target}.")

            target_slug, target_group, target_label = _target_info(target)
            key_cols = _key_cols(loaded)
            all_leads = sorted(
                {
                    int(v)
                    for df in loaded.values()
                    for v in pd.to_numeric(df["lead_time_h"], errors="coerce").dropna().unique()
                    if 1 <= int(v) <= int(horizon)
                }
            )
            missing_leads = sorted(expected_leads - set(all_leads))
            if missing_leads:
                warning_rows.append(
                    {
                        "split": split,
                        "target": target,
                        "severity": "warning",
                        "message": f"Missing lead hours in raw inputs: {','.join('h' + str(x) for x in missing_leads)}",
                    }
                )

            for lead in all_leads:
                lead_loaded = {
                    m.key: loaded[m.key].loc[pd.to_numeric(loaded[m.key]["lead_time_h"], errors="coerce").eq(float(lead))].copy()
                    for m in models
                }
                lead_qs = set.intersection(*(set(qmaps[m.key]) for m in models))
                if not lead_qs:
                    raise ValueError(f"No common quantile grid for split={split}, target={target}, lead=h{lead}.")
                common_qcols = {m.key: {q: qmaps[m.key][q] for q in sorted(lead_qs)} for m in models}
                valid = {m.key: _valid_frame(lead_loaded[m.key], common_qcols[m.key], key_cols) for m in models}
                common_keys = set.intersection(*(_key_tuples(valid[m.key], key_cols) for m in models))
                if not common_keys:
                    warning_rows.append(
                        {
                            "split": split,
                            "target": target,
                            "severity": "warning",
                            "message": f"No common valid rows for lead h{lead}.",
                        }
                    )
                    continue
                key_df = pd.DataFrame(list(common_keys), columns=key_cols)
                quantiles_used = ",".join(_qcol(q) for q in sorted(lead_qs))
                for model in models:
                    original_rows = int(len(lead_loaded[model.key]))
                    valid_rows = int(len(valid[model.key]))
                    retained_rows = int(len(common_keys))
                    dropped_rows = int(valid_rows - retained_rows)
                    row_rows.append(
                        {
                            "split": split,
                            "target": target,
                            "target_group": target_group,
                            "lead_time_h": int(lead),
                            "model": model.key,
                            "model_label": model.label,
                            "original_rows": original_rows,
                            "valid_rows": valid_rows,
                            "retained_common_rows": retained_rows,
                            "dropped_rows": dropped_rows,
                            "retained_share": retained_rows / valid_rows if valid_rows else float("nan"),
                            "quantiles_available": ",".join(_qcol(q) for q in sorted(qmaps[model.key])),
                            "quantiles_used": quantiles_used,
                            "row_intersection_key": ",".join(["split", "target", *key_cols]),
                            "eval_origin_start_utc": eval_origin_start.isoformat() if eval_origin_start is not None else "",
                            "eval_origin_end_utc": eval_origin_end.isoformat() if eval_origin_end is not None else "",
                        }
                    )
                    eval_df = valid[model.key].merge(key_df, on=key_cols, how="inner").sort_values(key_cols)
                    values = _metrics_for_frame(eval_df, common_qcols[model.key])
                    metric_rows.append(
                        {
                            "model": model.key,
                            "model_label": model.label,
                            "split": split,
                            "target": target,
                            "target_label": target_label,
                            "target_slug": target_slug,
                            "target_group": target_group,
                            "lead_time_h": int(lead),
                            "n_obs": retained_rows,
                            "n_valid_rows": retained_rows,
                            "quantiles_used": quantiles_used,
                            **values,
                        }
                    )

    metrics = sort_target_frame(pd.DataFrame(metric_rows), target_col="target", extra_cols=["split", "lead_time_h", "model_label"])
    row_counts = sort_target_frame(pd.DataFrame(row_rows), target_col="target", extra_cols=["split", "lead_time_h", "model_label"])
    warnings = pd.DataFrame(warning_rows, columns=["split", "target", "severity", "message"])
    range_summary = build_range_summary(metrics)
    return {"metrics": metrics, "range_summary": range_summary, "row_counts": row_counts, "warnings": warnings}


def build_range_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    d = metrics.copy()
    d["lead_range"] = d["lead_time_h"].map(assign_lead_range)
    d = d[d["lead_range"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for keys, part in d.groupby(["split", "target", "target_label", "target_group", "target_slug", "lead_range", "model", "model_label"], sort=False):
        split, target, target_label, target_group, target_slug, lead_range, model, model_label = keys
        weights = pd.to_numeric(part["n_obs"], errors="coerce").to_numpy(dtype=float)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        row = {
            "split": split,
            "target": target,
            "target_label": target_label,
            "target_group": target_group,
            "target_slug": target_slug,
            "lead_range": lead_range,
            "model": model,
            "model_label": model_label,
            "n_obs": int(np.nansum(weights)),
        }
        for metric in MAIN_METRICS:
            vals = pd.to_numeric(part[metric], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(vals) & (weights > 0)
            row[metric] = float(np.average(vals[mask], weights=weights[mask])) if mask.any() else float("nan")
        rows.append(row)
    return sort_target_frame(pd.DataFrame(rows), target_col="target", extra_cols=["split", "lead_range", "model_label"])


def build_thesis_range_table(range_summary: pd.DataFrame, *, split: str) -> pd.DataFrame:
    d = range_summary.loc[range_summary["split"] == split].copy()
    if d.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    target_col = "target" if "target" in d.columns else "target_label"
    d = sort_target_frame(d, target_col=target_col, extra_cols=["lead_range", "model_label"])
    for (target_label, lead_range), part in d.groupby(["target_label", "lead_range"], sort=False):
        values = {
            str(label): float(pd.to_numeric(group["mean_pinball_loss"], errors="coerce").mean())
            for label, group in part.groupby("model_label")
            if pd.to_numeric(group["mean_pinball_loss"], errors="coerce").notna().any()
        }
        best = min(values, key=values.get) if values else ""
        rows.append(
            {
                "target": target_label,
                "_sort_target": str(part["target"].iloc[0]) if "target" in part.columns else target_label,
                "lead_range": LEAD_RANGE_LABELS.get(str(lead_range), str(lead_range)),
                "RLQR": values.get("RLQR", np.nan),
                "XGB": values.get("XGB", np.nan),
                "TFT": values.get("TFT", np.nan),
                "best_model": best,
                "n_obs": int(part.groupby("model_label")["n_obs"].first().min()) if not part.empty else 0,
            }
        )
    order = {name: i for i, (name, _, _) in enumerate(LEAD_RANGES)}
    label_order = {LEAD_RANGE_LABELS[name]: idx for name, idx in order.items()}
    table = pd.DataFrame(rows)
    table["_range_order"] = table["lead_range"].map(label_order)
    table = sort_target_frame(table, target_col="_sort_target", extra_cols=["_range_order"])
    return table.drop(columns=["_sort_target", "_range_order"]).reset_index(drop=True)


def _slug_for_group(group: Any) -> str:
    s = str(group)
    for slug, label in [
        ("da_price", "DA price"),
        ("afrr_capacity_price", "aFRR capacity price"),
        ("afrr_activation_price", "aFRR activation price"),
        ("afrr_activation_rate", "aFRR activation rate"),
    ]:
        if s == label:
            return slug
    return s.lower().replace(" ", "_")


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


def _fmt_num(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "-"
    return f"{x:.4f}" if np.isfinite(x) else "-"


def write_latex_range_table(table: pd.DataFrame, *, out_dir: Path, split: str) -> Path | None:
    if table.empty:
        return None
    headers = ["Target", "Lead range", "RLQR", "XGB", "TFT", "Best model"]
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \begin{tabular}{@{}llrrrl@{}}",
        r"        \toprule",
        "        " + " & ".join(r"\textbf{" + _latex_escape(h) + "}" for h in headers) + r" \\",
        r"        \midrule",
    ]
    previous_group = None
    for _, row in table.iterrows():
        group = row["target"]
        vals = [
            r"\textbf{" + _latex_escape(group) + "}" if group != previous_group else "",
            _latex_escape(row["lead_range"]),
            _fmt_num(row["RLQR"]),
            _fmt_num(row["XGB"]),
            _fmt_num(row["TFT"]),
            _latex_escape(row["best_model"]),
        ]
        lines.append("        " + " & ".join(vals) + r" \\")
        previous_group = group
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \caption{Per-lead range mean pinball loss on the test split. Values are unweighted and computed on the common valid row intersection. Lower values indicate better probabilistic forecast performance.}",
            r"    \label{tab:per_lead_range_summary}",
            r"\end{table}",
            "",
        ]
    )
    path = out_dir / "latex" / f"per_lead_range_summary_{split}.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _figure_metric(metrics: pd.DataFrame, *, out_dir: Path, split: str, target_slug: str, metric: str, skip_pdf: bool = False) -> list[Path]:
    import matplotlib.pyplot as plt

    d = metrics.loc[(metrics["split"] == split) & (metrics["target_slug"] == target_slug)].copy()
    if d.empty or metric not in d.columns:
        return []
    apply_geo_style()
    targets = ordered_unique(d["target_label"].dropna().unique())
    n = max(1, len(targets))
    fig, axes = plt.subplots(n, 1, figsize=(10, max(3.2, 2.8 * n)), sharex=True)
    if n == 1:
        axes = [axes]
    group_label = str(d["target_group"].iloc[0])
    for ax, target_label in zip(axes, targets):
        panel = d[d["target_label"] == target_label].copy()
        for model in sorted(panel["model"].dropna().unique(), key=model_sort_key):
            mg = panel[panel["model"].eq(model)]
            mg = mg.sort_values("lead_time_h")
            show_markers = metric != "mean_pinball_loss"
            ax.plot(
                mg["lead_time_h"],
                mg[metric],
                marker="o" if show_markers else None,
                markersize=3.8 if show_markers else 0,
                linewidth=1.8,
                label=str(mg["model_label"].iloc[0]),
                color=get_model_color(str(mg["model"].iloc[0])),
            )
        ax.set_title(thesis_titlecase(str(target_label)))
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_xlim(1, max(48, int(pd.to_numeric(d["lead_time_h"], errors="coerce").max())))
        ax.set_xticks([1, 8, 16, 24, 32, 40, 48])
        ax.legend(ncol=3, loc="upper left")
    axes[-1].set_xlabel("Lead hour h1-h48")
    fig.suptitle(thesis_titlecase(f"{group_label}: {METRIC_LABELS.get(metric, metric)} by lead hour (lower is better)"), y=1.01)
    fig.tight_layout()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = f"per_lead_{'pinball' if metric == 'mean_pinball_loss' else metric}_{GROUP_FILE_STEMS[target_slug]}"
    paths = [fig_dir / f"{stem}.png"]
    if not skip_pdf:
        paths.append(fig_dir / f"{stem}.pdf")
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


def _figure_relative_pinball(metrics: pd.DataFrame, *, out_dir: Path, split: str, target_slug: str, skip_pdf: bool = False) -> list[Path]:
    import matplotlib.pyplot as plt

    d = metrics.loc[(metrics["split"] == split) & (metrics["target_slug"] == target_slug)].copy()
    if d.empty:
        return []
    pivot = (
        d.pivot_table(
            index=["target_label", "lead_time_h"],
            columns="model_label",
            values="mean_pinball_loss",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if "RLQR" not in pivot.columns:
        return []
    pivot = pivot[np.isfinite(pd.to_numeric(pivot["RLQR"], errors="coerce")) & (pd.to_numeric(pivot["RLQR"], errors="coerce").abs() > 1e-12)].copy()
    if pivot.empty:
        return []
    apply_geo_style()
    targets = ordered_unique(pivot["target_label"].dropna().unique())
    n = max(1, len(targets))
    fig, axes = plt.subplots(n, 1, figsize=(10, max(3.2, 2.8 * n)), sharex=True)
    if n == 1:
        axes = [axes]
    group_label = str(d["target_group"].iloc[0])
    for ax, target_label in zip(axes, targets):
        panel = pivot[pivot["target_label"] == target_label].copy().sort_values("lead_time_h")
        ax.axhline(1.0, color=get_model_color("linear"), linewidth=1.0, linestyle="--", label="RLQR baseline")
        for model_label, model_key in [("XGB", "xgb"), ("TFT", "tft")]:
            if model_label not in panel.columns:
                continue
            rel = pd.to_numeric(panel[model_label], errors="coerce") / pd.to_numeric(panel["RLQR"], errors="coerce")
            ax.plot(
                panel["lead_time_h"],
                rel,
                marker="o",
                markersize=3.8,
                linewidth=1.8,
                label=model_label,
                color=get_model_color(model_key),
            )
        ax.set_title(thesis_titlecase(str(target_label)))
        ax.set_ylabel("Pinball loss / RLQR")
        ax.set_xlim(1, max(48, int(pd.to_numeric(pivot["lead_time_h"], errors="coerce").max())))
        ax.set_xticks([1, 8, 16, 24, 32, 40, 48])
        ax.legend(ncol=3, loc="upper left")
    axes[-1].set_xlabel("Lead hour h1-h48")
    fig.suptitle(thesis_titlecase(f"{group_label}: mean pinball loss relative to RLQR by lead hour (below 1 is better)"), y=1.01)
    fig.tight_layout()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = f"per_lead_relative_pinball_{GROUP_FILE_STEMS[target_slug]}"
    paths = [fig_dir / f"{stem}.png"]
    if not skip_pdf:
        paths.append(fig_dir / f"{stem}.pdf")
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


def write_outputs(outputs: dict[str, pd.DataFrame], *, out_dir: Path, split: str, structured_out_dir: Path | None = None, skip_pdf: bool = False) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "latex").mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    metrics = outputs["metrics"]
    range_summary = outputs["range_summary"]
    row_counts = outputs["row_counts"]
    warnings = outputs["warnings"]
    table = build_thesis_range_table(range_summary, split=split)

    for path, df in [
        (out_dir / "per_lead_metrics.csv", metrics),
        (out_dir / f"per_lead_metrics_{split}.csv", metrics.loc[metrics["split"] == split]),
        (out_dir / f"per_lead_range_summary_{split}.csv", table),
        (out_dir / f"per_lead_range_summary_detail_{split}.csv", range_summary.loc[range_summary["split"] == split]),
        (out_dir / f"per_lead_row_counts_{split}.csv", row_counts.loc[row_counts["split"] == split]),
        (out_dir / "per_lead_warnings.csv", warnings),
    ]:
        df.to_csv(path, index=False)
        paths.append(path)

    tex = write_latex_range_table(table, out_dir=out_dir, split=split)
    if tex is not None:
        paths.append(tex)

    for target_slug in GROUP_ORDER:
        for metric in ["mean_pinball_loss", "mae_p50", "rmse_p50"]:
            paths.extend(_figure_metric(metrics, out_dir=out_dir, split=split, target_slug=target_slug, metric=metric, skip_pdf=skip_pdf))
        paths.extend(_figure_relative_pinball(metrics, out_dir=out_dir, split=split, target_slug=target_slug, skip_pdf=skip_pdf))

    manifest = {
        "description": "RQ1 4.1.3 per-lead-hour forecast benchmark outputs.",
        "split": split,
        "main_metric": "mean_pinball_loss",
        "secondary_metrics": ["mae_p50", "rmse_p50"],
        "lead_ranges": [{"name": name, "from": lo, "to": hi} for name, lo, hi in LEAD_RANGES],
        "row_intersection": "common valid row intersection across compared models per split, target, target_time_utc and lead_time_h; forecast_time_utc is included if available",
        "excluded": ["calibration_reliability", "interval_coverage_vs_nominal", "gate_specific", "tail_spike", "simulation_backtest"],
        "outputs": [str(p) for p in paths],
    }
    manifest_path = out_dir / "per_lead_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths.append(manifest_path)

    if structured_out_dir is not None:
        paths.extend(_mirror_structured(paths, root_out_dir=out_dir, structured_out_dir=structured_out_dir))
    return paths


def _mirror_structured(paths: list[Path], *, root_out_dir: Path, structured_out_dir: Path) -> list[Path]:
    mirrored: list[Path] = []
    for src in paths:
        if not src.exists() or src == structured_out_dir:
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
    p = argparse.ArgumentParser(description="Build final RQ1 4.1.3 per-lead-hour benchmark outputs.")
    p.add_argument("--benchmark-root", default="artifacts")
    p.add_argument("--benchmark-dir", default=None)
    p.add_argument("--out-dir", default="artifacts/rq1_ml_model_benchmark/_raw_outputs/shared")
    p.add_argument("--structured-out-dir", default="artifacts/rq1_ml_model_benchmark/_raw_outputs/4_1_3_per_lead")
    p.add_argument("--split", default="test", help="Main thesis/reporting split. Defaults to test.")
    p.add_argument(
        "--splits",
        default="test",
        help="Comma-separated splits to load/export. Defaults to test only; --split selects the main reported split.",
    )
    p.add_argument("--models", default="tft,xgboost,linear", help="Models to compare. Defaults to all RQ1 models: TFT, XGB and RLQR.")
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--eval-origin-start", default=DEFAULT_EVAL_ORIGIN_START_UTC, help="Inclusive forecast-origin lower bound for final RQ1 evaluation. Empty string disables the lower bound.")
    p.add_argument("--eval-origin-end", default=DEFAULT_EVAL_ORIGIN_END_UTC, help="Inclusive forecast-origin upper bound for final RQ1 evaluation. Empty string disables the upper bound.")
    p.add_argument("--no-structured-copy", action="store_true")
    p.add_argument("--skip-pdf", action="store_true", help="Do not render PDF figures; PNG and LaTeX outputs are still generated.")
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
    outputs = build_per_lead_outputs(
        benchmark_dir=benchmark_dir,
        models=models,
        splits=splits,
        horizon=int(args.horizon),
        eval_origin_start=_parse_utc_bound(args.eval_origin_start),
        eval_origin_end=_parse_utc_bound(args.eval_origin_end),
    )
    structured_out_dir = None if args.no_structured_copy else Path(args.structured_out_dir)
    paths = write_outputs(outputs, out_dir=Path(args.out_dir), split=args.split, structured_out_dir=structured_out_dir, skip_pdf=bool(args.skip_pdf))
    print("[OK] Built RQ1 4.1.3 per-lead-hour outputs.")
    print(f"[OK] benchmark_dir={benchmark_dir}")
    for path in paths:
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
