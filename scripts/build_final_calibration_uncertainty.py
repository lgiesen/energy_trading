#!/usr/bin/env python3
"""Build RQ1 calibration and uncertainty-quality outputs.

This script evaluates probabilistic forecast reliability, interval sharpness,
and quantile crossing on common valid row intersections across compared models.
It does not train models, run simulations, or repair/reorder quantiles before
evaluation.
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

from energy_trading.visualization.style import THESIS_PALETTE, apply_geo_style, get_model_color, thesis_titlecase
from energy_trading.evaluation.rq1_target_order import MODEL_LABELS, model_sort_key, ordered_unique, sort_target_frame, target_sort_key


MODEL_KEYS = {
    "tft": ("tft", "TFT"),
    "xgb": ("xgb", "XGB"),
    "xgboost": ("xgb", "XGB"),
    "linear": ("linear", "RLQR"),
    "rlqr": ("linear", "RLQR"),
}

TARGET_GROUPS = {
    "pred_da_price": ("DA price", "DA price"),
    "target_da_price": ("DA price", "DA price"),
    "pred_afrr_capacity_price_pos": ("aFRR capacity price", "aFRR capacity price +"),
    "target_afrr_capacity_price_pos": ("aFRR capacity price", "aFRR capacity price +"),
    "pred_afrr_capacity_price_neg": ("aFRR capacity price", "aFRR capacity price -"),
    "target_afrr_capacity_price_neg": ("aFRR capacity price", "aFRR capacity price -"),
    "pred_afrr_activation_price_pos": ("aFRR activation price", "aFRR activation price +"),
    "target_afrr_activation_price_pos": ("aFRR activation price", "aFRR activation price +"),
    "target_afrr_activation_price_vwap_pos": ("aFRR activation price", "aFRR activation price +"),
    "pred_afrr_activation_price_neg": ("aFRR activation price", "aFRR activation price -"),
    "target_afrr_activation_price_neg": ("aFRR activation price", "aFRR activation price -"),
    "target_afrr_activation_price_vwap_neg": ("aFRR activation price", "aFRR activation price -"),
    "pred_afrr_activation_rate_pos": ("aFRR activation rate", "aFRR activation rate +"),
    "target_afrr_activation_rate_pos": ("aFRR activation rate", "aFRR activation rate +"),
    "pred_afrr_activation_rate_neg": ("aFRR activation rate", "aFRR activation rate -"),
    "target_afrr_activation_rate_neg": ("aFRR activation rate", "aFRR activation rate -"),
}

ACTIVATION_RELIABILITY_AGGREGATES = (
    ("afrr_activation_price", "aFRR activation price"),
    ("afrr_activation_rate", "aFRR activation rate"),
)

REQUIRED_INTERVAL = (0.10, 0.90)
INTERVAL_PAIRS = [(0.30, 0.70), (0.10, 0.90), (0.05, 0.95), (0.01, 0.99)]
OPTIONAL_PAIRS = [(0.05, 0.95), (0.01, 0.99)]
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
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(ModelSpec(canonical, label))
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
                f"No benchmark directory with diagnostics/joined_predictions found under {benchmark_root}."
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


def _target_info(target: str) -> tuple[str, str]:
    return TARGET_GROUPS.get(str(target), ("Other", str(target).replace("_", " ")))


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


def _stacked_header(label: str) -> str:
    words = [word for word in str(label).split() if word]
    body = r"\\".join(_latex_escape(word) for word in words)
    return r"\textbf{\shortstack{" + body + r"}}"


def _read_joined(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    required = {"target_time_utc", "lead_time_h", "y_true", "p50"}
    missing = required - set(df.columns)
    if missing:
        if "p50" in missing:
            raise ValueError(f"Missing required p50 quantile column in {path}.")
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df["target_time_utc"] = pd.to_datetime(df["target_time_utc"], utc=True, errors="coerce")
    if "snapshot_time_utc" in df.columns:
        df["snapshot_time_utc"] = pd.to_datetime(df["snapshot_time_utc"], utc=True, errors="coerce")
    elif "forecast_time_utc" in df.columns:
        df["snapshot_time_utc"] = pd.to_datetime(df["forecast_time_utc"], utc=True, errors="coerce")
    else:
        df["snapshot_time_utc"] = df["target_time_utc"] - pd.to_timedelta(df["lead_time_h"], unit="h")
    df["lead_time_h"] = pd.to_numeric(df["lead_time_h"], errors="coerce")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["p50"] = pd.to_numeric(df["p50"], errors="coerce")
    return df


def _parse_utc_bound(raw: str | None) -> pd.Timestamp | None:
    if raw is None or str(raw).strip() == "":
        return None
    return pd.to_datetime(raw, utc=True)


def _apply_forecast_origin_window(df: pd.DataFrame, *, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    if start is None and end is None:
        return df
    origin = pd.to_datetime(df["snapshot_time_utc"], utc=True, errors="coerce")
    mask = origin.notna()
    if start is not None:
        mask &= origin.ge(start)
    if end is not None:
        mask &= origin.le(end)
    return df.loc[mask].copy()


def _key_cols(loaded: dict[str, pd.DataFrame]) -> list[str]:
    has_snapshot = all("snapshot_time_utc" in df.columns and df["snapshot_time_utc"].notna().any() for df in loaded.values())
    return ["snapshot_time_utc", "target_time_utc", "lead_time_h"] if has_snapshot else ["target_time_utc", "lead_time_h"]


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


def _filter_common(df: pd.DataFrame, common_keys: set[tuple[Any, ...]], key_cols: list[str]) -> pd.DataFrame:
    key_df = pd.DataFrame(list(common_keys), columns=key_cols)
    out = df.merge(key_df, on=key_cols, how="inner")
    return out.sort_values(key_cols).reset_index(drop=True)


def _quantile_coverage(
    df: pd.DataFrame,
    *,
    model: ModelSpec,
    split: str,
    target: str,
    qcols: dict[float, str],
) -> pd.DataFrame:
    target_group, target_label = _target_info(target)
    y = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    p50 = pd.to_numeric(df["p50"], errors="coerce").to_numpy(dtype=float)
    p50_bias = float(np.mean(p50 - y))
    rows = []
    for q in sorted(qcols):
        pred = pd.to_numeric(df[qcols[q]], errors="coerce").to_numpy(dtype=float)
        coverage = float(np.mean(y <= pred))
        err = coverage - float(q)
        rows.append(
            {
                "model": model.key,
                "model_label": model.label,
                "split": split,
                "target": target,
                "target_label": target_label,
                "target_group": target_group,
                "quantile": float(q),
                "quantile_col": qcols[q],
                "empirical_coverage": coverage,
                "calibration_error": err,
                "abs_calibration_error": abs(err),
                "p50_bias": p50_bias,
                "n_obs": int(len(df)),
            }
        )
    return pd.DataFrame(rows)


def _interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    width = hi - lo
    score = width.copy()
    below = y < lo
    above = y > hi
    score[below] = width[below] + (2.0 / alpha) * (lo[below] - y[below])
    score[above] = width[above] + (2.0 / alpha) * (y[above] - hi[above])
    return float(np.mean(score))


def _interval_metrics(
    df: pd.DataFrame,
    *,
    model: ModelSpec,
    split: str,
    target: str,
    qcols: dict[float, str],
) -> pd.DataFrame:
    target_group, target_label = _target_info(target)
    y = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    rows = []
    for qlo, qhi in INTERVAL_PAIRS:
        if qlo not in qcols or qhi not in qcols:
            continue
        lo = pd.to_numeric(df[qcols[qlo]], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(df[qcols[qhi]], errors="coerce").to_numpy(dtype=float)
        inside = (lo <= y) & (y <= hi)
        width = hi - lo
        nominal = float(qhi - qlo)
        alpha = float(1.0 - nominal)
        coverage = float(np.mean(inside))
        rows.append(
            {
                "model": model.key,
                "model_label": model.label,
                "split": split,
                "target": target,
                "target_label": target_label,
                "target_group": target_group,
                "interval": f"{_qcol(qlo)}-{_qcol(qhi)}",
                "q_low": float(qlo),
                "q_high": float(qhi),
                "nominal_interval_coverage": nominal,
                "interval_coverage": coverage,
                "interval_coverage_error": coverage - nominal,
                "interval_width_mean": float(np.mean(width)),
                "interval_width_median": float(np.median(width)),
                "interval_score": _interval_score(y, lo, hi, alpha=alpha),
                "n_obs": int(len(df)),
            }
        )
    return pd.DataFrame(rows)


def _crossing_metrics(
    df: pd.DataFrame,
    *,
    model: ModelSpec,
    split: str,
    target: str,
    qcols: dict[float, str],
) -> dict[str, Any]:
    target_group, target_label = _target_info(target)
    ordered = [qcols[q] for q in sorted(qcols)]
    arr = df[ordered].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    diffs = np.diff(arr, axis=1) if arr.shape[1] >= 2 else np.empty((len(df), 0))
    violations = diffs < 0.0
    row_has = np.any(violations, axis=1) if violations.size else np.zeros(len(df), dtype=bool)
    mags = np.where(violations, -diffs, 0.0) if violations.size else np.zeros((len(df), 0), dtype=float)
    pos = mags[mags > 0]
    return {
        "model": model.key,
        "model_label": model.label,
        "split": split,
        "target": target,
        "target_label": target_label,
        "target_group": target_group,
        "quantiles_used": ",".join(_qcol(q) for q in sorted(qcols)),
        "n_obs": int(len(df)),
        "crossing_any": bool(row_has.any()),
        "crossing_rate": float(np.mean(row_has)) if len(df) else float("nan"),
        "num_crossings": int(violations.sum()) if violations.size else 0,
        "mean_crossing_magnitude": float(np.mean(pos)) if pos.size else 0.0,
        "max_crossing_magnitude": float(np.max(pos)) if pos.size else 0.0,
    }


def build_calibration_outputs(
    *,
    benchmark_dir: Path,
    models: list[ModelSpec],
    splits: list[str],
    eval_origin_start: pd.Timestamp | None,
    eval_origin_end: pd.Timestamp | None,
) -> dict[str, pd.DataFrame]:
    joined_dir = benchmark_dir / "diagnostics" / "joined_predictions"
    files: dict[tuple[str, str, str], Path] = {}
    for path in joined_dir.glob("*.parquet"):
        parsed = _parse_joined_name(path)
        if parsed is not None:
            files[parsed] = path

    targets = sorted({target for _, split, target in files if split in set(splits)}, key=target_sort_key)
    if not targets:
        raise FileNotFoundError(f"No joined prediction parquet files for splits={splits} in {joined_dir}.")

    coverage_parts: list[pd.DataFrame] = []
    interval_parts: list[pd.DataFrame] = []
    crossing_rows: list[dict[str, Any]] = []
    row_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []

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
                    raise ValueError(f"Missing required p50 quantile for model={model.key}, split={split}, target={target}.")

            common_qs = set.intersection(*(set(qmaps[m.key]) for m in models))
            if 0.10 not in common_qs or 0.90 not in common_qs:
                raise ValueError(f"Missing required common p10/p90 interval for split={split}, target={target}.")
            for qlo, qhi in OPTIONAL_PAIRS:
                if qlo not in common_qs or qhi not in common_qs:
                    warning_rows.append(
                        {
                            "split": split,
                            "target": target,
                            "severity": "warning",
                            "message": f"Optional interval {_qcol(qlo)}-{_qcol(qhi)} missing from common quantile grid.",
                        }
                    )

            key_cols = _key_cols(loaded)
            common_qcols = {m.key: {q: qmaps[m.key][q] for q in sorted(common_qs)} for m in models}
            valid = {m.key: _valid_frame(loaded[m.key], common_qcols[m.key], key_cols) for m in models}
            common_keys = set.intersection(*(_key_tuples(valid[m.key], key_cols) for m in models))
            if not common_keys:
                raise ValueError(f"Common valid row intersection is empty for split={split}, target={target}.")
            retained = int(len(common_keys))
            for model in models:
                original = int(len(loaded[model.key]))
                valid_n = int(len(valid[model.key]))
                dropped = int(valid_n - retained)
                row_rows.append(
                    {
                        "split": split,
                        "target": target,
                        "target_group": _target_info(target)[0],
                        "model": model.key,
                        "model_label": model.label,
                        "original_rows": original,
                        "valid_rows": valid_n,
                        "retained_common_rows": retained,
                        "dropped_rows": dropped,
                        "retained_share": retained / valid_n if valid_n else float("nan"),
                        "quantiles_available": ",".join(_qcol(q) for q in sorted(qmaps[model.key])),
                        "quantiles_used": ",".join(_qcol(q) for q in sorted(common_qs)),
                        "row_intersection_key": ",".join(["split", "target", *key_cols]),
                        "eval_origin_start_utc": eval_origin_start.isoformat() if eval_origin_start is not None else "",
                        "eval_origin_end_utc": eval_origin_end.isoformat() if eval_origin_end is not None else "",
                    }
                )
                eval_df = _filter_common(valid[model.key], common_keys, key_cols)
                coverage_parts.append(
                    _quantile_coverage(eval_df, model=model, split=split, target=target, qcols=common_qcols[model.key])
                )
                interval_parts.append(
                    _interval_metrics(eval_df, model=model, split=split, target=target, qcols=common_qcols[model.key])
                )
                crossing_rows.append(
                    _crossing_metrics(eval_df, model=model, split=split, target=target, qcols=common_qcols[model.key])
                )

    coverage = pd.concat(coverage_parts, ignore_index=True) if coverage_parts else pd.DataFrame()
    interval = pd.concat(interval_parts, ignore_index=True) if interval_parts else pd.DataFrame()
    crossing = pd.DataFrame(crossing_rows)
    row_counts = pd.DataFrame(row_rows)
    warnings = pd.DataFrame(warning_rows, columns=["split", "target", "severity", "message"])
    summary = build_summary(coverage, interval, crossing)
    return {
        "quantile_coverage": coverage,
        "interval_coverage_width": interval,
        "quantile_crossing": crossing,
        "row_counts": row_counts,
        "warnings": warnings,
        "summary": summary,
    }


def build_summary(coverage: pd.DataFrame, interval: pd.DataFrame, crossing: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame()
    cal = (
        coverage.groupby(["model", "model_label", "split", "target", "target_label", "target_group"], as_index=False)
        .agg(
            mean_abs_calibration_error=("abs_calibration_error", "mean"),
            signed_mean_calibration_error=("calibration_error", "mean"),
            p50_bias=("p50_bias", "first"),
            n_obs=("n_obs", "first"),
            quantiles_used=("quantile_col", lambda s: ",".join(str(x) for x in s)),
        )
    )
    p50 = coverage.loc[np.isclose(pd.to_numeric(coverage["quantile"], errors="coerce"), 0.50)]
    if not p50.empty:
        cal = cal.merge(
            p50[["model", "split", "target", "empirical_coverage", "calibration_error"]].rename(
                columns={"empirical_coverage": "p50_empirical_coverage", "calibration_error": "p50_calibration_error"}
            ),
            on=["model", "split", "target"],
            how="left",
        )
    main_int = interval.loc[interval["interval"].eq("p10-p90")].copy() if not interval.empty else pd.DataFrame()
    if not main_int.empty:
        keep = [
            "model",
            "split",
            "target",
            "interval_coverage",
            "interval_coverage_error",
            "interval_width_mean",
            "interval_width_median",
            "interval_score",
        ]
        cal = cal.merge(
            main_int[keep].rename(
                columns={
                    "interval_coverage": "p10_p90_interval_coverage",
                    "interval_coverage_error": "p10_p90_interval_coverage_error",
                    "interval_width_mean": "p10_p90_interval_width_mean",
                    "interval_width_median": "p10_p90_interval_width_median",
                    "interval_score": "p10_p90_interval_score",
                }
            ),
            on=["model", "split", "target"],
            how="left",
        )
    if not crossing.empty:
        cal = cal.merge(
            crossing[
                [
                    "model",
                    "split",
                    "target",
                    "crossing_rate",
                    "num_crossings",
                    "mean_crossing_magnitude",
                    "max_crossing_magnitude",
                ]
            ],
            on=["model", "split", "target"],
            how="left",
        )
    return sort_target_frame(cal, target_col="target", extra_cols=["split", "model_label"])


def _format_num(v: Any, digits: int = 4) -> str:
    try:
        x = float(v)
    except Exception:
        return "-"
    if not np.isfinite(x):
        return "-"
    return f"{x:.{digits}f}"


def _format_num_bold_if_best(v: Any, best: float, digits: int = 4, tol: float = 1e-12) -> str:
    formatted = _format_num(v, digits=digits)
    if formatted == "-":
        return formatted
    try:
        x = float(v)
    except Exception:
        return formatted
    if np.isfinite(x) and np.isfinite(float(best)) and abs(x - float(best)) <= float(tol):
        return r"\textbf{" + formatted + "}"
    return formatted


def _format_pp(v: Any, digits: int = 2) -> str:
    try:
        x = float(v)
    except Exception:
        return "-"
    if not np.isfinite(x):
        return "-"
    return f"{x * 100.0:.{digits}f} pp"


def _format_pp_bold_if_best(v: Any, best: float, digits: int = 2, tol: float = 1e-12) -> str:
    formatted = _format_pp(v, digits=digits)
    if formatted == "-":
        return formatted
    try:
        x = float(v)
    except Exception:
        return formatted
    if np.isfinite(x) and np.isfinite(float(best)) and abs(x - float(best)) <= float(tol):
        return r"\textbf{" + formatted + "}"
    return formatted


def _main_issue_for_target(part: pd.DataFrame) -> str:
    crossing = pd.to_numeric(part.get("crossing_rate"), errors="coerce")
    if crossing.notna().any() and float(crossing.max()) > 1e-9:
        return "quantile crossing"
    best = part.sort_values("mean_abs_calibration_error", ascending=True).iloc[0]
    cov_err = float(best.get("p10_p90_interval_coverage_error", np.nan))
    mace = float(best.get("mean_abs_calibration_error", np.nan))
    if np.isfinite(cov_err) and cov_err < -0.05:
        return "undercoverage"
    if np.isfinite(cov_err) and cov_err > 0.05:
        return "overcoverage"
    if np.isfinite(mace) and mace <= 0.03:
        return "none"
    return "mixed"


def build_main_summary_table(summary: pd.DataFrame, *, split: str) -> pd.DataFrame:
    d = summary.loc[summary["split"] == split].copy()
    if d.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for target, part in d.groupby("target", sort=True):
        part = part.copy()
        best = part.sort_values("mean_abs_calibration_error", ascending=True).iloc[0]
        vals = {
            str(r["model_label"]): float(r["mean_abs_calibration_error"])
            for _, r in part.iterrows()
            if pd.notna(r.get("mean_abs_calibration_error"))
        }
        rows.append(
            {
                "target": target,
                "target_label": str(best["target_label"]),
                "RLQR_MACE": vals.get("RLQR", np.nan),
                "XGB_MACE": vals.get("XGB", np.nan),
                "TFT_MACE": vals.get("TFT", np.nan),
                "best_calibrated": str(best["model_label"]),
                "p10_p90_coverage": float(best.get("p10_p90_interval_coverage", np.nan)),
                "main_issue": _main_issue_for_target(part),
            }
        )
    return sort_target_frame(pd.DataFrame(rows), target_col="target_label")


def write_latex_summary(summary: pd.DataFrame, *, out_dir: Path, split: str) -> Path | None:
    d = build_main_summary_table(summary, split=split)
    if d.empty:
        return None
    headers = ["Target", "RLQR MACE", "XGB MACE", "TFT MACE", "Best calibrated", "p10-p90 coverage"]
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \begin{tabular}{@{}lrrrlr@{}}",
        r"        \toprule",
        "        " + " & ".join(_stacked_header(h) for h in headers) + r" \\",
        r"        \midrule",
    ]
    for _, row in d.iterrows():
        mace_values = [
            float(row.get(col))
            for col in ("RLQR_MACE", "XGB_MACE", "TFT_MACE")
            if pd.notna(row.get(col)) and np.isfinite(float(row.get(col)))
        ]
        best_mace = min(mace_values) if mace_values else float("nan")
        vals = [
            _latex_escape(row["target_label"]),
            _format_pp_bold_if_best(row.get("RLQR_MACE"), best_mace),
            _format_pp_bold_if_best(row.get("XGB_MACE"), best_mace),
            _format_pp_bold_if_best(row.get("TFT_MACE"), best_mace),
            _latex_escape(row["best_calibrated"]),
            _format_pp(row.get("p10_p90_coverage")),
        ]
        lines.append("        " + " & ".join(vals) + r" \\")
    mean_mace_by_col: dict[str, float] = {}
    for col in ("RLQR_MACE", "XGB_MACE", "TFT_MACE"):
        values = pd.to_numeric(d[col], errors="coerce") if col in d.columns else pd.Series(dtype=float)
        finite_values = values[np.isfinite(values)]
        mean_mace_by_col[col] = float(finite_values.mean()) if not finite_values.empty else float("nan")
    finite_means = [v for v in mean_mace_by_col.values() if np.isfinite(v)]
    best_mean_mace = min(finite_means) if finite_means else float("nan")
    best_mean_model = "-"
    for label, col in (("RLQR", "RLQR_MACE"), ("XGB", "XGB_MACE"), ("TFT", "TFT_MACE")):
        value = mean_mace_by_col[col]
        if np.isfinite(value) and np.isfinite(best_mean_mace) and abs(value - best_mean_mace) <= 1e-12:
            best_mean_model = label
            break
    lines.append(r"        \midrule")
    lines.append(
        "        "
        + " & ".join(
            [
                "Mean MACE",
                _format_pp_bold_if_best(mean_mace_by_col["RLQR_MACE"], best_mean_mace),
                _format_pp_bold_if_best(mean_mace_by_col["XGB_MACE"], best_mean_mace),
                _format_pp_bold_if_best(mean_mace_by_col["TFT_MACE"], best_mean_mace),
                _latex_escape(best_mean_model),
                "-",
            ]
        )
        + r" \\"
    )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \caption{MACE and p10-p90 coverage in percentage points for each target variable.}",
            r"    \label{tab:calibration_summary_test}",
            r"\end{table}",
            "",
        ]
    )
    path = out_dir / "latex" / f"rq1_4_1_2_calibration_summary_{split}.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_latex_quantile_coverage_appendix(coverage: pd.DataFrame, *, out_dir: Path, split: str) -> Path | None:
    d = coverage.loc[coverage["split"] == split].copy()
    if d.empty:
        return None
    d = sort_target_frame(d, target_col="target", extra_cols=["quantile", "model_label"])
    rows: list[dict[str, Any]] = []
    for (target, q), part in d.groupby(["target", "quantile"], sort=False):
        vals = {
            str(r["model_label"]): float(r["empirical_coverage"])
            for _, r in part.iterrows()
            if pd.notna(r.get("empirical_coverage"))
        }
        abs_errs = {
            str(r["model_label"]): abs(float(r["calibration_error"]))
            for _, r in part.iterrows()
            if pd.notna(r.get("calibration_error"))
        }
        best = min(abs_errs, key=abs_errs.get) if abs_errs else ""
        label = str(part["target_label"].iloc[0])
        rows.append(
            {
                "target_label": label,
                "quantile": float(q),
                "RLQR": vals.get("RLQR", np.nan),
                "XGB": vals.get("XGB", np.nan),
                "TFT": vals.get("TFT", np.nan),
                "ideal": float(q),
                "best_error": float(abs_errs[best]) if best else np.nan,
                "best_model": best,
            }
        )
    table = sort_target_frame(pd.DataFrame(rows), target_col="target_label", extra_cols=["quantile"])
    headers = ["Target", "Quantile", "RLQR empirical coverage", "XGB empirical coverage", "TFT empirical coverage", "Ideal coverage", "Best model abs. error"]
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \begin{tabular}{@{}lrrrrrl@{}}",
        r"        \toprule",
        "        " + " & ".join(r"\textbf{" + _latex_escape(h) + "}" for h in headers) + r" \\",
        r"        \midrule",
    ]
    for _, row in table.iterrows():
        best_error = f"{row['best_model']} ({_format_num(row['best_error'])})" if str(row.get("best_model", "")) else "-"
        vals = [
            _latex_escape(row["target_label"]),
            _format_num(row["quantile"], 2),
            _format_num(row["RLQR"]),
            _format_num(row["XGB"]),
            _format_num(row["TFT"]),
            _format_num(row["ideal"], 2),
            _latex_escape(best_error),
        ]
        lines.append("        " + " & ".join(vals) + r" \\")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \caption{Detailed empirical quantile coverage on the test split.}",
            r"    \label{tab:calibration_quantile_coverage_test_appendix}",
            r"\end{table}",
            "",
        ]
    )
    path = out_dir / "latex" / f"rq1_4_1_2_calibration_quantile_coverage_{split}_appendix.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_latex_interval_quality_appendix(interval: pd.DataFrame, *, out_dir: Path, split: str) -> Path | None:
    d = interval.loc[interval["split"] == split].copy()
    if d.empty:
        return None
    d = sort_target_frame(d, target_col="target_label", extra_cols=["interval", "model_label"])
    headers = ["Target", "Interval", "Model", "Nominal coverage", "Empirical coverage", "Coverage error", "Mean width", "Median width", "Interval score"]
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \begin{tabular}{@{}lllrrrrrr@{}}",
        r"        \toprule",
        "        " + " & ".join(r"\textbf{" + _latex_escape(h) + "}" for h in headers) + r" \\",
        r"        \midrule",
    ]
    for _, row in d.iterrows():
        vals = [
            _latex_escape(row["target_label"]),
            _latex_escape(row["interval"]),
            _latex_escape(row["model_label"]),
            _format_num(row["nominal_interval_coverage"]),
            _format_num(row["interval_coverage"]),
            _format_num(row["interval_coverage_error"]),
            _format_num(row["interval_width_mean"]),
            _format_num(row["interval_width_median"]),
            _format_num(row["interval_score"]),
        ]
        lines.append("        " + " & ".join(vals) + r" \\")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \caption{Detailed interval coverage and sharpness diagnostics on the test split.}",
            r"    \label{tab:calibration_interval_quality_test_appendix}",
            r"\end{table}",
            "",
        ]
    )
    path = out_dir / "latex" / f"rq1_4_1_2_calibration_interval_quality_{split}_appendix.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_latex_crossing_appendix(crossing: pd.DataFrame, *, out_dir: Path, split: str) -> Path | None:
    d = crossing.loc[crossing["split"] == split].copy()
    if d.empty:
        return None
    d = sort_target_frame(d, target_col="target_label", extra_cols=["model_label"])
    headers = ["Target", "Model", "Crossing rate", "Mean crossing magnitude", "Max crossing magnitude"]
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \begin{tabular}{@{}llrrr@{}}",
        r"        \toprule",
        "        " + " & ".join(r"\textbf{" + _latex_escape(h) + "}" for h in headers) + r" \\",
        r"        \midrule",
    ]
    for _, row in d.iterrows():
        vals = [
            _latex_escape(row["target_label"]),
            _latex_escape(row["model_label"]),
            _format_num(row["crossing_rate"]),
            _format_num(row["mean_crossing_magnitude"]),
            _format_num(row["max_crossing_magnitude"]),
        ]
        lines.append("        " + " & ".join(vals) + r" \\")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \caption{Quantile crossing diagnostics on the test split.}",
            r"    \label{tab:calibration_quantile_crossing_test_appendix}",
            r"\end{table}",
            "",
        ]
    )
    path = out_dir / "latex" / f"rq1_4_1_2_calibration_quantile_crossing_{split}_appendix.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _group_panel_axes(groups: list[str]):
    import matplotlib.pyplot as plt

    n = max(1, len(groups))
    fig, axes = plt.subplots(n, 1, figsize=(10, max(3.2, 2.8 * n)), sharex=False)
    if n == 1:
        axes = [axes]
    return fig, axes


def aggregate_activation_reliability(coverage: pd.DataFrame) -> pd.DataFrame:
    """Merge positive and negative activation targets for reliability plots."""
    if coverage.empty:
        return pd.DataFrame()
    required = {"target_group", "model", "model_label", "quantile", "empirical_coverage", "n_obs"}
    if not required.issubset(coverage.columns):
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for aggregate_slug, aggregate_label in ACTIVATION_RELIABILITY_AGGREGATES:
        part = coverage.loc[coverage["target_group"].eq(aggregate_label)].copy()
        if part.empty:
            continue
        part["n_obs"] = pd.to_numeric(part["n_obs"], errors="coerce")
        part["empirical_coverage"] = pd.to_numeric(part["empirical_coverage"], errors="coerce")
        part = part.loc[part["n_obs"].notna() & part["empirical_coverage"].notna() & (part["n_obs"] > 0)].copy()
        if part.empty:
            continue
        part["_weighted_empirical_coverage"] = part["empirical_coverage"] * part["n_obs"]
        grouped = (
            part.groupby(["model", "model_label", "quantile"], as_index=False)
            .agg(
                weighted_empirical_coverage=("_weighted_empirical_coverage", "sum"),
                n_obs=("n_obs", "sum"),
            )
        )
        grouped["empirical_coverage"] = grouped["weighted_empirical_coverage"] / grouped["n_obs"]
        grouped["calibration_error"] = grouped["empirical_coverage"] - pd.to_numeric(grouped["quantile"], errors="coerce")
        grouped["target"] = aggregate_slug
        grouped["target_label"] = aggregate_label
        grouped["target_group"] = aggregate_label
        frames.append(grouped.drop(columns=["weighted_empirical_coverage"]))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["target_label", "model_label", "quantile"]).reset_index(drop=True)


def plot_reliability_by_target_group(coverage: pd.DataFrame, fig_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    if coverage.empty:
        return None
    apply_geo_style()
    d = (
        coverage.groupby(["target", "target_label", "target_group", "model", "model_label", "quantile"], as_index=False)
        .agg(empirical_coverage=("empirical_coverage", "mean"), n_obs=("n_obs", "sum"))
    )
    d = sort_target_frame(d, target_col="target", extra_cols=["model_label", "quantile"])
    targets = ordered_unique(d["target_label"].dropna().unique())
    fig, axes = _group_panel_axes(targets)
    fig.suptitle(thesis_titlecase("Empirical coverage vs nominal quantile"), y=1.01)
    for ax, target_label in zip(axes, targets):
        g = d[d["target_label"] == target_label]
        for model in sorted(g["model"].dropna().unique(), key=model_sort_key):
            mg = g[g["model"].eq(model)]
            mg = mg.sort_values("quantile")
            ax.plot(
                mg["quantile"],
                mg["empirical_coverage"],
                marker="o",
                label=str(mg["model_label"].iloc[0]),
                color=get_model_color(str(mg["model"].iloc[0])),
            )
        ax.plot([0, 1], [0, 1], linestyle="--", color=THESIS_PALETTE["neutral_dark"], label="Ideal")
        n_obs = pd.to_numeric(g["n_obs"], errors="coerce").dropna()
        n_label = int(n_obs.min()) if not n_obs.empty else 0
        ax.set_title(thesis_titlecase(f"{target_label} (n={n_label})"))
        ax.set_xlabel("Nominal quantile")
        ax.set_ylabel("Empirical coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ticks = np.array([0.10, 0.30, 0.50, 0.70, 0.90], dtype=float)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        pct = FuncFormatter(lambda value, _: f"{value * 100:.0f}%")
        ax.xaxis.set_major_formatter(pct)
        ax.yaxis.set_major_formatter(pct)
        ax.legend(ncol=4, loc="upper left")
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "rq1_4_1_2_calibration_reliability_by_target.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_reliability_activation_aggregates(coverage: pd.DataFrame, fig_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    d = aggregate_activation_reliability(coverage)
    if d.empty:
        return None
    apply_geo_style()
    targets = ordered_unique(d["target_label"].dropna().unique())
    fig, axes = _group_panel_axes(targets)
    fig.suptitle(thesis_titlecase("Aggregate empirical coverage vs nominal quantile"), y=1.01)
    for ax, target_label in zip(axes, targets):
        g = d[d["target_label"] == target_label]
        for model in sorted(g["model"].dropna().unique(), key=model_sort_key):
            mg = g[g["model"].eq(model)].sort_values("quantile")
            ax.plot(
                mg["quantile"],
                mg["empirical_coverage"],
                marker="o",
                label=str(mg["model_label"].iloc[0]),
                color=get_model_color(str(mg["model"].iloc[0])),
            )
        ax.plot([0, 1], [0, 1], linestyle="--", color=THESIS_PALETTE["neutral_dark"], label="Ideal")
        n_obs = pd.to_numeric(g["n_obs"], errors="coerce").dropna()
        n_label = int(n_obs.min()) if not n_obs.empty else 0
        ax.set_title(thesis_titlecase(f"{target_label} (+/- merged, n={n_label})"))
        ax.set_xlabel("Nominal quantile")
        ax.set_ylabel("Empirical coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ticks = np.array([0.10, 0.30, 0.50, 0.70, 0.90], dtype=float)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        pct = FuncFormatter(lambda value, _: f"{value * 100:.0f}%")
        ax.xaxis.set_major_formatter(pct)
        ax.yaxis.set_major_formatter(pct)
        ax.legend(ncol=4, loc="upper left")
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "rq1_4_1_2_calibration_reliability_activation_aggregates.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_interval_coverage_by_target_group(interval: pd.DataFrame, fig_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    if interval.empty:
        return None
    apply_geo_style()
    d = (
        interval.groupby(["target_group", "model", "model_label", "nominal_interval_coverage"], as_index=False)
        .agg(interval_coverage=("interval_coverage", "mean"), n_obs=("n_obs", "sum"))
        .sort_values(["target_group", "model_label", "nominal_interval_coverage"])
    )
    if d.empty:
        return None
    groups = ordered_unique(d["target_group"].dropna().unique(), group=True)
    fig, axes = _group_panel_axes(groups)
    fig.suptitle(thesis_titlecase("Interval coverage vs nominal interval coverage"), y=1.01)
    ticks = np.array([0.40, 0.80, 0.90, 0.98], dtype=float)
    pct = FuncFormatter(lambda value, _: f"{value * 100:.0f}%")
    for ax, group in zip(axes, groups):
        g = d[d["target_group"] == group]
        for model in sorted(g["model"].dropna().unique(), key=model_sort_key):
            mg = g[g["model"].eq(model)]
            mg = mg.sort_values("nominal_interval_coverage")
            ax.plot(
                mg["nominal_interval_coverage"],
                mg["interval_coverage"],
                marker="o",
                label=str(mg["model_label"].iloc[0]),
                color=get_model_color(str(mg["model"].iloc[0])),
            )
        ax.plot([0, 1], [0, 1], linestyle="--", color=THESIS_PALETTE["neutral_dark"], label="Ideal")
        ax.set_title(thesis_titlecase(f"{group} (n={int(g['n_obs'].sum())})"))
        ax.set_xlabel("Nominal interval coverage")
        ax.set_ylabel("Empirical interval coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.xaxis.set_major_formatter(pct)
        ax.yaxis.set_major_formatter(pct)
        ax.legend(ncol=4, loc="upper left")
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "rq1_4_1_2_calibration_interval_coverage_by_target_group.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_interval_width_by_target_group(interval: pd.DataFrame, fig_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    if interval.empty:
        return None
    apply_geo_style()
    d = (
        interval.groupby(["target_group", "model", "model_label", "interval"], as_index=False)
        .agg(interval_width_mean=("interval_width_mean", "mean"), n_obs=("n_obs", "sum"))
    )
    order = ["p30-p70", "p10-p90", "p05-p95", "p01-p99"]
    groups = ordered_unique(d["target_group"].dropna().unique(), group=True)
    fig, axes = _group_panel_axes(groups)
    for ax, group in zip(axes, groups):
        g = d[d["target_group"] == group].copy()
        intervals = [x for x in order if x in set(g["interval"])]
        x = np.arange(len(intervals), dtype=float)
        width = 0.22
        models = sorted(g["model"].dropna().unique(), key=model_sort_key)
        for i, model in enumerate(models):
            mg = g[g["model"].eq(model)]
            vals = [float(mg.loc[mg["interval"] == it, "interval_width_mean"].mean()) for it in intervals]
            ax.bar(x + (i - 1) * width, vals, width=width, label=str(mg["model_label"].iloc[0]), color=get_model_color(str(mg["model"].iloc[0])))
        ax.set_xticks(x)
        ax.set_xticklabels(intervals)
        ax.set_title(thesis_titlecase(f"{group} (n={int(g['n_obs'].sum())})"))
        ax.set_ylabel("Mean interval width")
        ax.legend(ncol=4, loc="upper left")
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "rq1_4_1_2_calibration_interval_width_by_target_group.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_crossing_by_target_group(crossing: pd.DataFrame, fig_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    if crossing.empty:
        return None
    apply_geo_style()
    d = crossing.groupby(["target_group", "model", "model_label"], as_index=False).agg(crossing_rate=("crossing_rate", "mean"), n_obs=("n_obs", "sum"))
    groups = ordered_unique(d["target_group"].dropna().unique(), group=True)
    x = np.arange(len(groups), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 4.8))
    models = sorted(d["model"].dropna().unique(), key=model_sort_key)
    for i, model in enumerate(models):
        mg = d[d["model"].eq(model)]
        vals = [float(mg.loc[mg["target_group"] == g, "crossing_rate"].mean()) for g in groups]
        ax.bar(x + (i - 1) * width, vals, width=width, label=str(mg["model_label"].iloc[0]), color=get_model_color(str(mg["model"].iloc[0])))
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel("Quantile crossing rate")
    ax.set_title(thesis_titlecase("Quantile crossing rate by target group"))
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "rq1_4_1_2_calibration_quantile_crossing_by_target_group.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_calibration_error_heatmap(coverage: pd.DataFrame, fig_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    if coverage.empty:
        return None
    apply_geo_style()
    d = (
        coverage.groupby(["target_group", "model_label", "quantile"], as_index=False)
        .agg(calibration_error=("calibration_error", "mean"))
    )
    d["_target_group_order"] = d["target_group"].map(lambda x: target_sort_key(x)[0])
    d["_model_order"] = d["model_label"].map(lambda x: model_sort_key(x)[0])
    d = d.sort_values(["_target_group_order", "_model_order", "quantile"])
    d["row"] = d["target_group"] + " | " + d["model_label"]
    pivot = d.pivot_table(index="row", columns="quantile", values="calibration_error", aggfunc="mean")
    row_order = d["row"].drop_duplicates().tolist()
    pivot = pivot.reindex(row_order)
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-0.25, vmax=0.25)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([_qcol(float(q)) for q in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(thesis_titlecase("Calibration error by target group, model and quantile"))
    fig.colorbar(im, ax=ax, label="Empirical coverage - nominal quantile")
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "rq1_4_1_2_calibration_error_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _split_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if df.empty or "split" not in df.columns:
        return df.copy()
    return df.loc[df["split"] == split].copy()


def write_legacy_flat_aliases(outputs: dict[str, pd.DataFrame], *, out_dir: Path, source_dir: Path, split: str) -> list[Path]:
    """Mirror current structured outputs to thesis-facing flat calibration paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    csv_aliases = {
        "quantile_coverage": "calibration_quantile_coverage",
        "interval_coverage_width": "calibration_interval_coverage_width",
        "quantile_crossing": "calibration_quantile_crossing",
        "summary": "calibration_summary",
    }
    for key, stem in csv_aliases.items():
        df = outputs[key]
        path = out_dir / f"{stem}.csv"
        df.to_csv(path, index=False)
        paths.append(path)
        path_split = out_dir / f"{stem}_{split}.csv"
        _split_frame(df, split).to_csv(path_split, index=False)
        paths.append(path_split)

    for key, name in [("row_counts", "calibration_row_counts.csv"), ("warnings", "calibration_warnings.csv")]:
        path = out_dir / name
        outputs[key].to_csv(path, index=False)
        paths.append(path)

    copy_aliases = {
        source_dir / "latex" / f"rq1_4_1_2_calibration_summary_{split}.tex": out_dir / "latex" / f"calibration_summary_{split}.tex",
        source_dir / "latex" / f"rq1_4_1_2_calibration_quantile_coverage_{split}_appendix.tex": out_dir / "latex" / f"calibration_quantile_coverage_{split}_appendix.tex",
        source_dir / "latex" / f"rq1_4_1_2_calibration_interval_quality_{split}_appendix.tex": out_dir / "latex" / f"calibration_interval_quality_{split}_appendix.tex",
        source_dir / "latex" / f"rq1_4_1_2_calibration_quantile_crossing_{split}_appendix.tex": out_dir / "latex" / f"calibration_quantile_crossing_{split}_appendix.tex",
        source_dir / "figures" / "rq1_4_1_2_calibration_reliability_by_target.png": out_dir / "figures" / "calibration_reliability_by_target.png",
        source_dir / "figures" / "rq1_4_1_2_calibration_interval_coverage_by_target_group.png": out_dir / "figures" / "calibration_interval_coverage_by_target_group.png",
        source_dir / "figures" / "rq1_4_1_2_calibration_interval_width_by_target_group.png": out_dir / "figures" / "calibration_interval_width_by_target_group.png",
        source_dir / "figures" / "rq1_4_1_2_calibration_quantile_crossing_by_target_group.png": out_dir / "figures" / "calibration_quantile_crossing_by_target_group.png",
        source_dir / "figures" / "rq1_4_1_2_calibration_error_heatmap.png": out_dir / "figures" / "calibration_error_heatmap.png",
        source_dir / "figures" / "rq1_4_1_2_calibration_reliability_activation_aggregates.png": out_dir / "figures" / "calibration_reliability_activation_aggregates.png",
    }
    for src, dst in copy_aliases.items():
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        paths.append(dst)

    manifest = {
        "description": "Legacy flat aliases for RQ1 section 4.1.2 calibration and uncertainty-quality outputs.",
        "split": split,
        "source_dir": str(source_dir),
        "outputs": [str(p) for p in paths],
    }
    manifest_path = out_dir / "calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths.append(manifest_path)
    return paths


def write_outputs(
    outputs: dict[str, pd.DataFrame],
    *,
    out_dir: Path,
    split: str,
    legacy_flat_out_dir: Path | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    name_map = {
        "quantile_coverage": "rq1_4_1_2_calibration_quantile_coverage",
        "interval_coverage_width": "rq1_4_1_2_calibration_interval_coverage_width",
        "quantile_crossing": "rq1_4_1_2_calibration_quantile_crossing",
        "summary": "rq1_4_1_2_calibration_summary",
    }
    for key, stem in name_map.items():
        df = outputs[key]
        path = csv_dir / f"{stem}.csv"
        df.to_csv(path, index=False)
        paths.append(path)
        path_split = csv_dir / f"{stem}_{split}.csv"
        df.loc[df["split"] == split].to_csv(path_split, index=False)
        paths.append(path_split)
    for key, stem in [("row_counts", "rq1_4_1_2_calibration_row_counts"), ("warnings", "rq1_4_1_2_calibration_warnings")]:
        path = csv_dir / f"{stem}.csv"
        outputs[key].to_csv(path, index=False)
        paths.append(path)

    latex = write_latex_summary(outputs["summary"], out_dir=out_dir, split=split)
    if latex is not None:
        paths.append(latex)
    for latex_appendix in [
        write_latex_quantile_coverage_appendix(outputs["quantile_coverage"], out_dir=out_dir, split=split),
        write_latex_interval_quality_appendix(outputs["interval_coverage_width"], out_dir=out_dir, split=split),
        write_latex_crossing_appendix(outputs["quantile_crossing"], out_dir=out_dir, split=split),
    ]:
        if latex_appendix is not None:
            paths.append(latex_appendix)

    fig_dir = out_dir / "figures"
    coverage_split = _split_frame(outputs["quantile_coverage"], split)
    interval_split = _split_frame(outputs["interval_coverage_width"], split)
    crossing_split = _split_frame(outputs["quantile_crossing"], split)
    for p in [
        plot_reliability_by_target_group(coverage_split, fig_dir),
        plot_reliability_activation_aggregates(coverage_split, fig_dir),
        plot_interval_coverage_by_target_group(interval_split, fig_dir),
        plot_interval_width_by_target_group(interval_split, fig_dir),
        plot_crossing_by_target_group(crossing_split, fig_dir),
        plot_calibration_error_heatmap(coverage_split, fig_dir),
    ]:
        if p is not None:
            paths.append(p)

    manifest = {
        "description": "RQ1 calibration and uncertainty-quality outputs.",
        "split": split,
        "definitions": {
            "coverage_q": "mean(y_true <= y_pred_q)",
            "calibration_error_q": "coverage_q - q",
            "interval_coverage": "mean(y_pred_low <= y_true <= y_pred_high)",
            "interval_width_mean": "mean(y_pred_high - y_pred_low)",
            "interval_score": "central interval score; lower is better conditional on interval and scale",
            "crossing": "reported on original quantile predictions; no monotonic repair is applied",
        },
        "outputs": [str(p) for p in paths],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths.append(manifest_path)
    if legacy_flat_out_dir is not None:
        paths.extend(write_legacy_flat_aliases(outputs, out_dir=legacy_flat_out_dir, source_dir=out_dir, split=split))
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build final RQ1 calibration and uncertainty-quality outputs.")
    p.add_argument("--benchmark-root", default="artifacts")
    p.add_argument("--benchmark-dir", default=None)
    p.add_argument("--out-dir", default="artifacts/benchmark/rq1_ml_model_benchmark/_raw_outputs/4_1_2_calibration_uncertainty")
    p.add_argument("--split", default="test", help="Main thesis/reporting split. Defaults to test.")
    p.add_argument(
        "--splits",
        default="test",
        help="Comma-separated splits to load/export. Defaults to test only; --split selects the main reported split.",
    )
    p.add_argument("--models", default="tft,xgboost,linear", help="Models to compare. Defaults to all RQ1 models: TFT, XGB and RLQR.")
    p.add_argument("--eval-origin-start", default=DEFAULT_EVAL_ORIGIN_START_UTC, help="Inclusive forecast-origin lower bound for final RQ1 evaluation. Empty string disables the lower bound.")
    p.add_argument("--eval-origin-end", default=DEFAULT_EVAL_ORIGIN_END_UTC, help="Inclusive forecast-origin upper bound for final RQ1 evaluation. Empty string disables the upper bound.")
    p.add_argument("--legacy-flat-out-dir", default="artifacts/benchmark/rq1_ml_model_benchmark/_raw_outputs/calibration")
    p.add_argument("--no-legacy-flat-aliases", action="store_true")
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
    outputs = build_calibration_outputs(
        benchmark_dir=benchmark_dir,
        models=models,
        splits=splits,
        eval_origin_start=_parse_utc_bound(args.eval_origin_start),
        eval_origin_end=_parse_utc_bound(args.eval_origin_end),
    )
    legacy_flat_out_dir = None if args.no_legacy_flat_aliases else Path(args.legacy_flat_out_dir)
    paths = write_outputs(outputs, out_dir=Path(args.out_dir), split=args.split, legacy_flat_out_dir=legacy_flat_out_dir)
    print("[OK] Built RQ1 calibration and uncertainty-quality outputs.")
    print(f"[OK] benchmark_dir={benchmark_dir}")
    for path in paths:
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
