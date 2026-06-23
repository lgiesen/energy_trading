#!/usr/bin/env python3
"""Build final RQ1 tail/spike behavior benchmark outputs.

This script evaluates existing joined forecast benchmark predictions only. It
defines realized tail/spike regimes, enforces common valid rows across models,
and reports unweighted probabilistic/point metrics for RQ1 diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
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

from energy_trading.evaluation.style import THESIS_PALETTE, apply_geo_style, get_model_color, thesis_titlecase
from energy_trading.evaluation.rq1_target_order import model_sort_key, ordered_model_labels, sort_target_frame, target_sort_key


QCOL_RE = re.compile(r"^p(\d{1,2})$", re.IGNORECASE)
EPSILON_DEFAULT = 1e-9
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

REGIME_LABELS = {
    "normal": "Normal",
    "da_positive_spike_top5": "Positive spike top 5%",
    "da_negative_spike_bottom5": "Negative spike bottom 5%",
    "afrr_capacity_price_high_tail_top5": "High tail top 5%",
    "afrr_activation_price_abs_tail_top5": "Abs. tail top 5%",
    "afrr_activation_rate_high_tail_top5": "High activation-rate tail top 5%",
    "activation_nonzero": "Activation nonzero",
    "activation_zero_or_nearzero": "Activation zero / near-zero",
    "high_volatility_week": "High-volatility week",
    "spike_week": "Spike week",
}

MAIN_ISSUES = {
    "normal": "Reference regime",
    "da_positive_spike_top5": "Positive price spike",
    "da_negative_spike_bottom5": "Negative price spike",
    "afrr_capacity_price_high_tail_top5": "Capacity price high tail",
    "afrr_activation_price_abs_tail_top5": "Large activation price",
    "afrr_activation_rate_high_tail_top5": "High activation rate",
    "activation_nonzero": "Activation occurrence",
    "activation_zero_or_nearzero": "No/near-zero activation",
    "high_volatility_week": "Volatile week",
    "spike_week": "Event-heavy week",
}

TARGET_GROUP_ORDER = [
    "DA price",
    "aFRR capacity price",
    "aFRR activation price",
    "aFRR activation rate",
]

MODEL_ORDER = {"linear": 0, "xgb": 1, "tft": 2}
REGIME_ORDER = {regime: i for i, regime in enumerate(REGIME_LABELS)}

MAIN_REGIMES_BY_TARGET_LABEL = {
    "DA price": [
        "normal",
        "da_positive_spike_top5",
        "da_negative_spike_bottom5",
        "high_volatility_week",
        "spike_week",
    ],
    "aFRR capacity price +": [
        "normal",
        "afrr_capacity_price_high_tail_top5",
        "high_volatility_week",
        "spike_week",
    ],
    "aFRR capacity price -": [
        "normal",
        "afrr_capacity_price_high_tail_top5",
        "high_volatility_week",
        "spike_week",
    ],
    "aFRR activation price +": [
        "normal",
        "afrr_activation_price_abs_tail_top5",
        "high_volatility_week",
        "spike_week",
    ],
    "aFRR activation price -": [
        "normal",
        "afrr_activation_price_abs_tail_top5",
        "high_volatility_week",
        "spike_week",
    ],
    "aFRR activation rate +": [
        "activation_zero_or_nearzero",
        "activation_nonzero",
        "high_volatility_week",
        "spike_week",
    ],
    "aFRR activation rate -": [
        "activation_zero_or_nearzero",
        "activation_nonzero",
        "high_volatility_week",
        "spike_week",
    ],
}

MAIN_REGIME_ORDER = {
    target_label: {regime: i for i, regime in enumerate(regimes)}
    for target_label, regimes in MAIN_REGIMES_BY_TARGET_LABEL.items()
}
MAIN_PRICE_CAPACITY_TARGETS = ["DA price", "aFRR capacity price +", "aFRR capacity price -"]
MAIN_ACTIVATION_TARGETS = [
    "aFRR activation price +",
    "aFRR activation price -",
    "aFRR activation rate +",
    "aFRR activation rate -",
]

DA_TARGET = "target_da_price"
CAPACITY_PRICE_TARGETS = {"target_afrr_capacity_price_pos", "target_afrr_capacity_price_neg"}
ACTIVATION_PRICE_TARGETS = {"target_afrr_activation_price_vwap_pos", "target_afrr_activation_price_vwap_neg"}
ACTIVATION_RATE_TARGETS = {"target_afrr_activation_rate_pos", "target_afrr_activation_rate_neg"}
ALL_TARGETS = {DA_TARGET, *CAPACITY_PRICE_TARGETS, *ACTIVATION_PRICE_TARGETS, *ACTIVATION_RATE_TARGETS}


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


def canonical_target(target: Any) -> str:
    return TARGET_ALIASES.get(str(target), str(target))


def _target_info(target: str) -> tuple[str, str]:
    return TARGET_GROUPS.get(canonical_target(target), ("Other", str(target).replace("_", " ")))


def _clean_regime_label(regime: Any) -> str:
    return REGIME_LABELS.get(str(regime), str(regime).replace("_", " ").title())


def _main_regime_order_key(group: Any, regime: Any) -> tuple[int, str]:
    group_s = str(group)
    regime_s = str(regime)
    return MAIN_REGIME_ORDER.get(group_s, {}).get(regime_s, 99), _clean_regime_label(regime_s)


def _main_issue(regime: Any) -> str:
    return MAIN_ISSUES.get(str(regime), "Conditional robustness")


def _target_group_sort_key(group: Any) -> tuple[int, str]:
    label = str(group)
    try:
        return TARGET_GROUP_ORDER.index(label), label
    except ValueError:
        return len(TARGET_GROUP_ORDER), label


def _target_group_slug(group: Any) -> str:
    special = {
        "DA price": "da_price",
        "aFRR capacity price": "capacity_price",
        "aFRR activation price": "activation_price",
        "aFRR activation rate": "activation_rate",
    }
    if str(group) in special:
        return special[str(group)]
    return (
        str(group)
        .lower()
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace(" ", "_")
        .replace("/", "_")
    )


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
        raise FileNotFoundError(f"Missing joined predictions directory: {joined}")
    return out


def _read_joined(path: Path, *, source_target: str, derive_forecast_time: bool) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    missing = {"target_time_utc", "lead_time_h", "y_true"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df["target_time_utc"] = pd.to_datetime(df["target_time_utc"], utc=True, errors="coerce")
    if "forecast_time_utc" in df.columns:
        df["forecast_time_utc"] = pd.to_datetime(df["forecast_time_utc"], utc=True, errors="coerce")
    elif "snapshot_time_utc" in df.columns:
        df["forecast_time_utc"] = pd.to_datetime(df["snapshot_time_utc"], utc=True, errors="coerce")
    elif derive_forecast_time:
        df["forecast_time_utc"] = df["target_time_utc"] - pd.to_timedelta(pd.to_numeric(df["lead_time_h"], errors="coerce"), unit="h")
    else:
        raise ValueError(f"{path} is missing forecast_time_utc/snapshot_time_utc.")
    df["lead_time_h"] = pd.to_numeric(df["lead_time_h"], errors="coerce")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    if "p50" not in df.columns and "predicted_value" in df.columns:
        df["p50"] = pd.to_numeric(df["predicted_value"], errors="coerce")
    if "p50" not in df.columns:
        raise ValueError(f"{path} must contain p50 or predicted_value.")
    df["target"] = canonical_target(source_target)
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


def _pinball_values(y: np.ndarray, pred: np.ndarray, q: float) -> np.ndarray:
    err = y - pred
    return np.maximum(q * err, (q - 1.0) * err)


def _lead_summary(leads: pd.Series) -> str:
    vals = sorted({int(x) for x in pd.to_numeric(leads, errors="coerce").dropna().tolist()})
    if not vals:
        return ""
    out: list[str] = []
    start = prev = vals[0]
    for value in vals[1:]:
        if value == prev + 1:
            prev = value
            continue
        out.append(f"h{start}" if start == prev else f"h{start}-h{prev}")
        start = prev = value
    out.append(f"h{start}" if start == prev else f"h{start}-h{prev}")
    return ",".join(out)


def _week_start_utc(times: pd.Series) -> pd.Series:
    ts = pd.to_datetime(times, utc=True, errors="coerce")
    return ts.dt.tz_convert(None).dt.to_period("W").dt.start_time.dt.tz_localize("UTC")


def _valid_frame(df: pd.DataFrame, qcols: dict[float, str]) -> pd.DataFrame:
    cols = list(dict.fromkeys(["forecast_time_utc", "target_time_utc", "lead_time_h", "y_true", "p50", *[qcols[q] for q in sorted(qcols)]]))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    out = df[cols].copy()
    for col in ["y_true", "p50", *[qcols[q] for q in sorted(qcols)]]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    mask = out[["forecast_time_utc", "target_time_utc", "lead_time_h"]].notna().all(axis=1)
    for col in ["y_true", "p50", *[qcols[q] for q in sorted(qcols)]]:
        mask &= np.isfinite(out[col].to_numpy(dtype=float))
    return out.loc[mask].copy()


def _key_tuples(df: pd.DataFrame) -> set[tuple[Any, ...]]:
    return set(map(tuple, df[["forecast_time_utc", "target_time_utc", "lead_time_h"]].itertuples(index=False, name=None)))


def choose_threshold_source(splits: list[str], requested_split: str) -> tuple[str, list[str]]:
    if requested_split == "test" and "val" in splits:
        return "validation", ["val"]
    if "train" in splits and "val" in splits:
        return "train_validation", ["train", "val"]
    return "test_conditional" if requested_split == "test" else requested_split, [requested_split]


def resolve_threshold_source(splits: list[str], requested_split: str, threshold_source: str) -> tuple[str, list[str]]:
    source = str(threshold_source).strip().lower()
    if source == "test":
        return "test_conditional", [requested_split]
    if source == "validation":
        if "val" not in splits:
            raise ValueError("--threshold-source validation requires --splits to include val.")
        return "validation", ["val"]
    if source == "train_validation":
        missing = [s for s in ["train", "val"] if s not in splits]
        if missing:
            raise ValueError(f"--threshold-source train_validation requires --splits to include {missing}.")
        return "train_validation", ["train", "val"]
    raise ValueError(f"Unsupported threshold source {threshold_source!r}.")


def _threshold(y: pd.Series, q: float) -> float:
    vals = pd.to_numeric(y, errors="coerce").dropna()
    return float(vals.quantile(q)) if not vals.empty else float("nan")


def compute_target_thresholds(
    reference_by_target: dict[str, pd.DataFrame],
    *,
    threshold_source: str,
    epsilon: float,
    source_splits: list[str] | None = None,
    selection_split: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_splits_s = ",".join(source_splits or [])
    for target, df in sorted(reference_by_target.items(), key=lambda item: target_sort_key(item[0])):
        y = pd.to_numeric(df["y_true"], errors="coerce")
        group, target_display = _target_info(target)
        common = {
            "target": target,
            "target_display": target_display,
            "target_group": group,
            "threshold_source": threshold_source,
            "source_splits": source_splits_s,
            "selection_split": selection_split or "",
        }
        if target == DA_TARGET:
            rows.extend(
                [
                    {**common, "regime": "da_positive_spike_top5", "threshold_name": "q95", "threshold_value": _threshold(y, 0.95)},
                    {**common, "regime": "da_negative_spike_bottom5", "threshold_name": "q05", "threshold_value": _threshold(y, 0.05)},
                ]
            )
        if target in CAPACITY_PRICE_TARGETS:
            rows.append(
                {
                    **common,
                    "regime": "afrr_capacity_price_high_tail_top5",
                    "threshold_name": "capacity_price_high_tail_q95",
                    "threshold_value": _threshold(y, 0.95),
                }
            )
        if target in ACTIVATION_PRICE_TARGETS:
            rows.append({**common, "regime": "afrr_activation_price_abs_tail_top5", "threshold_name": "abs_q95", "threshold_value": _threshold(y.abs(), 0.95)})
        if target in ACTIVATION_RATE_TARGETS:
            rows.extend(
                [
                    {
                        **common,
                        "regime": "afrr_activation_rate_high_tail_top5",
                        "threshold_name": "activation_rate_high_tail_q95",
                        "threshold_value": _threshold(y, 0.95),
                    },
                    {**common, "regime": "activation_nonzero", "threshold_name": "epsilon", "threshold_value": float(epsilon), "threshold_source": "domain"},
                    {**common, "regime": "activation_zero_or_nearzero", "threshold_name": "epsilon", "threshold_value": float(epsilon), "threshold_source": "domain"},
                ]
            )
    return pd.DataFrame(rows)


def regime_masks_for_target(df: pd.DataFrame, thresholds: pd.DataFrame, *, epsilon: float) -> dict[str, pd.Series]:
    if df.empty:
        return {}
    target = canonical_target(df["target"].iloc[0]) if "target" in df.columns else ""
    y = pd.to_numeric(df["y_true"], errors="coerce")
    masks: dict[str, pd.Series] = {}
    t = thresholds[thresholds["target"].eq(target)]
    for _, row in t.iterrows():
        regime = str(row["regime"])
        thr = float(row["threshold_value"])
        if regime == "afrr_activation_price_abs_tail_top5":
            masks[regime] = y.abs().ge(thr)
        elif regime == "afrr_capacity_price_high_tail_top5":
            masks[regime] = y.ge(thr)
        elif regime == "afrr_activation_rate_high_tail_top5":
            masks[regime] = y.gt(float(epsilon)) & y.ge(thr)
        elif regime == "da_positive_spike_top5":
            masks[regime] = y.ge(thr)
        elif regime == "da_negative_spike_bottom5":
            masks[regime] = y.le(thr)
        elif regime == "activation_nonzero":
            masks[regime] = y.gt(float(epsilon))
        elif regime == "activation_zero_or_nearzero":
            masks[regime] = y.le(float(epsilon))
    if masks:
        event_union = pd.concat(masks.values(), axis=1).any(axis=1)
        masks["normal"] = ~event_union
    else:
        masks["normal"] = pd.Series(True, index=df.index)
    return masks


def _spike_event_regimes_for_target(target: str) -> set[str]:
    target = canonical_target(target)
    if target == DA_TARGET:
        return {"da_positive_spike_top5", "da_negative_spike_bottom5"}
    if target in CAPACITY_PRICE_TARGETS:
        return {"afrr_capacity_price_high_tail_top5"}
    if target in ACTIVATION_PRICE_TARGETS:
        return {"afrr_activation_price_abs_tail_top5"}
    if target in ACTIVATION_RATE_TARGETS:
        return {"afrr_activation_rate_high_tail_top5"}
    return set()


def _spike_selection_rule_for_target(target: str) -> str:
    if canonical_target(target) in ACTIVATION_RATE_TARGETS:
        return "Top 10% weeks by target-specific high-activation-rate tail share and count, selected separately per split and target."
    return "Top 10% weeks by target-specific spike_share and spike_count, selected separately per split and target."


def select_high_volatility_weeks(base: pd.DataFrame, *, top_share: float = 0.10) -> pd.DataFrame:
    d = base.copy()
    d = sort_target_frame(d, target_col="target", extra_cols=["split"])
    d["week_start_utc"] = _week_start_utc(d["target_time_utc"])
    rows: list[pd.DataFrame] = []
    for (split, target), part in d.groupby(["split", "target"], sort=False):
        target_group, target_display = _target_info(str(target))
        weekly = part.groupby("week_start_utc", as_index=False).agg(
            volatility_score=("y_true", "std"),
            n_rows=("y_true", "size"),
        )
        if weekly.empty:
            continue
        n = max(1, int(np.ceil(len(weekly) * float(top_share))))
        top = weekly.sort_values(["volatility_score", "week_start_utc"], ascending=[False, True]).head(n).copy()
        top["split"] = split
        top["target"] = target
        top["target_display"] = target_display
        top["target_group"] = target_group
        top["selection_type"] = "high_volatility_week"
        top["selection_rank"] = np.arange(1, len(top) + 1)
        top["selection_scope"] = "target"
        top["selection_rule"] = "Top 10% weeks by weekly std(y_true), selected separately per split and target."
        rows.append(top)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def select_spike_weeks(
    base: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    epsilon: float,
    top_share: float = 0.10,
    warnings: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    parts = []
    base = sort_target_frame(base, target_col="target", extra_cols=["split"])
    for target, part in base.groupby("target", sort=False):
        masks = regime_masks_for_target(part, thresholds, epsilon=epsilon)
        spike_regimes = _spike_event_regimes_for_target(str(target))
        event_masks = [m for name, m in masks.items() if name in spike_regimes]
        if not event_masks:
            if warnings is not None:
                warnings.append(
                    {
                        "split": ",".join(sorted(map(str, part["split"].dropna().unique()))) if "split" in part.columns else "",
                        "target": target,
                        "regime": "spike_week",
                        "severity": "warning",
                        "message": "No target-specific spike threshold was available for the formal spike-event regimes; spike_week selection skipped for this target.",
                    }
                )
            continue
        threshold_part = thresholds.loc[thresholds["target"].eq(target) & thresholds["regime"].isin(spike_regimes)].copy()
        threshold_names = ",".join(str(x) for x in threshold_part["threshold_name"].dropna().unique()) if "threshold_name" in threshold_part.columns else ""
        finite_thresholds = pd.to_numeric(threshold_part.get("threshold_value", pd.Series(dtype=float)), errors="coerce").dropna()
        threshold_value = float(finite_thresholds.max()) if not finite_thresholds.empty else np.nan
        event = pd.concat(event_masks, axis=1).any(axis=1)
        p = part.copy()
        p["event"] = event.to_numpy(dtype=bool)
        p["threshold_name"] = threshold_names
        p["threshold_value"] = threshold_value
        parts.append(p)
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    d = sort_target_frame(d, target_col="target", extra_cols=["split"])
    d["week_start_utc"] = _week_start_utc(d["target_time_utc"])
    rows: list[pd.DataFrame] = []
    for (split, target), part in d.groupby(["split", "target"], sort=False):
        target_group, target_display = _target_info(str(target))
        weekly = part.groupby("week_start_utc", as_index=False).agg(
            spike_count=("event", "sum"),
            n_rows=("event", "size"),
            threshold_name=("threshold_name", "first"),
            threshold_value=("threshold_value", "max"),
        )
        weekly["spike_share"] = weekly["spike_count"] / weekly["n_rows"].replace(0, np.nan)
        n = max(1, int(np.ceil(len(weekly) * float(top_share))))
        top = weekly.sort_values(["spike_share", "spike_count", "week_start_utc"], ascending=[False, False, True]).head(n).copy()
        top["split"] = split
        top["target"] = target
        top["target_display"] = target_display
        top["target_group"] = target_group
        top["selection_type"] = "spike_week"
        top["selection_rank"] = np.arange(1, len(top) + 1)
        top["selection_scope"] = "target"
        top["selection_rule"] = _spike_selection_rule_for_target(str(target))
        rows.append(top)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _metrics(df: pd.DataFrame, qcols: dict[float, str]) -> dict[str, Any]:
    y = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    p50 = pd.to_numeric(df["p50"], errors="coerce").to_numpy(dtype=float)
    pinballs = [
        _pinball_values(y, pd.to_numeric(df[qcols[q]], errors="coerce").to_numpy(dtype=float), q)
        for q in sorted(qcols)
    ]
    out: dict[str, Any] = {
        "mean_pinball_loss": float(np.mean(np.vstack(pinballs))) if pinballs else float("nan"),
        "mae_p50": float(np.mean(np.abs(p50 - y))),
        "rmse_p50": float(np.sqrt(np.mean((p50 - y) ** 2))),
        "bias_p50": float(np.mean(p50 - y)),
        "median_absolute_error_p50": float(np.median(np.abs(p50 - y))),
        "n_obs": int(len(df)),
        "n_forecast_timestamps": int(df["forecast_time_utc"].nunique()),
        "n_target_timestamps": int(df["target_time_utc"].nunique()),
        "observed_lead_min": float(pd.to_numeric(df["lead_time_h"], errors="coerce").min()),
        "observed_lead_max": float(pd.to_numeric(df["lead_time_h"], errors="coerce").max()),
        "observed_leads": _lead_summary(df["lead_time_h"]),
    }
    for qlo, qhi in [(0.10, 0.90), (0.05, 0.95), (0.01, 0.99)]:
        if qlo in qcols and qhi in qcols:
            lo = pd.to_numeric(df[qcols[qlo]], errors="coerce").to_numpy(dtype=float)
            hi = pd.to_numeric(df[qcols[qhi]], errors="coerce").to_numpy(dtype=float)
            label = f"{_qcol(qlo)}_{_qcol(qhi)}"
            out[f"coverage_{label}"] = float(np.mean((lo <= y) & (y <= hi)))
            out[f"interval_width_{label}_mean"] = float(np.mean(hi - lo))
    return out


def _regime_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"regime": "normal", "definition": "Rows not belonging to a selected tail/spike regime for the same target."},
            {"regime": "da_positive_spike_top5", "definition": "target_da_price rows with y_true at or above 95th percentile threshold."},
            {"regime": "da_negative_spike_bottom5", "definition": "target_da_price rows with y_true at or below 5th percentile threshold."},
            {"regime": "afrr_capacity_price_high_tail_top5", "definition": "Capacity price rows with y_true at or above the target-specific 95th percentile threshold."},
            {"regime": "afrr_activation_price_abs_tail_top5", "definition": "activation price rows with abs(y_true) at or above abs 95th percentile threshold."},
            {"regime": "afrr_activation_rate_high_tail_top5", "definition": "Activation-rate rows with y_true at or above the target-specific 95th percentile threshold; used for spike-week selection, not as the activation occurrence regime."},
            {"regime": "activation_nonzero", "definition": "activation rate rows with y_true > epsilon."},
            {"regime": "activation_zero_or_nearzero", "definition": "activation rate rows with y_true <= epsilon."},
            {"regime": "high_volatility_week", "definition": "Rows in top 10% weeks by weekly realized standard deviation per split and target."},
            {"regime": "spike_week", "definition": "Rows in top 10% weeks by target-specific formal spike-event share per split and target."},
        ]
    )


def build_tail_spike_outputs(
    *,
    benchmark_dir: Path,
    models: list[ModelSpec],
    splits: list[str],
    main_split: str,
    epsilon: float,
    threshold_source: str = "test",
    derive_forecast_time: bool = True,
    eval_origin_start: pd.Timestamp | None = None,
    eval_origin_end: pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    joined_dir = benchmark_dir / "diagnostics" / "joined_predictions"
    files: dict[tuple[str, str, str], Path] = {}
    for path in sorted(joined_dir.glob("*.parquet")):
        parsed = _parse_joined_name(path)
        if parsed is not None:
            files[parsed] = path
    source_targets = sorted({target for _, split, target in files if split in set(splits) and canonical_target(target) in ALL_TARGETS}, key=target_sort_key)
    if not source_targets:
        raise FileNotFoundError(f"No supported joined prediction files for splits={splits} in {joined_dir}.")

    loaded: dict[tuple[str, str, str], pd.DataFrame] = {}
    qmaps: dict[tuple[str, str, str], dict[float, str]] = {}
    warnings: list[dict[str, Any]] = []
    base_parts: list[pd.DataFrame] = []
    for split in splits:
        for source_target in source_targets:
            canonical = canonical_target(source_target)
            for model in models:
                path = files.get((model.key, split, source_target))
                if path is None:
                    continue
                df = _apply_forecast_origin_window(
                    _read_joined(path, source_target=source_target, derive_forecast_time=derive_forecast_time),
                    start=eval_origin_start,
                    end=eval_origin_end,
                )
                loaded[(model.key, split, canonical)] = df
                qmaps[(model.key, split, canonical)] = _quantile_cols(df)
                if 0.50 not in qmaps[(model.key, split, canonical)]:
                    raise ValueError(f"Missing p50 for model={model.key}, split={split}, target={canonical}.")
            first_model = next((m for m in models if (m.key, split, canonical) in loaded), None)
            if first_model is not None:
                b = loaded[(first_model.key, split, canonical)][["forecast_time_utc", "target_time_utc", "lead_time_h", "y_true", "target"]].copy()
                b["split"] = split
                b["target"] = canonical
                b["target_group"] = _target_info(canonical)[0]
                base_parts.append(b)

    base = pd.concat(base_parts, ignore_index=True) if base_parts else pd.DataFrame()
    source_name, reference_splits = resolve_threshold_source(splits, main_split, threshold_source)
    if source_name == "test_conditional":
        warnings.append({"split": main_split, "target": "", "regime": "", "severity": "warning", "message": "Tail thresholds are test-conditional because --threshold-source defaults to test for this RQ1 thesis output."})
    ref = base[base["split"].isin(reference_splits)].copy()
    thresholds = compute_target_thresholds(
        {t: g for t, g in ref.groupby("target")},
        threshold_source=source_name,
        epsilon=epsilon,
        source_splits=reference_splits,
        selection_split=main_split,
    )
    for _, row in thresholds.loc[
        thresholds["regime"].eq("afrr_activation_rate_high_tail_top5")
        & pd.to_numeric(thresholds["threshold_value"], errors="coerce").le(float(epsilon))
    ].iterrows():
        warnings.append(
            {
                "split": main_split,
                "target": str(row.get("target", "")),
                "regime": "afrr_activation_rate_high_tail_top5",
                "severity": "warning",
                "message": "Activation-rate q95 is less than or equal to epsilon; spike mask uses y_true > epsilon and y_true >= q95.",
            }
        )
    warnings.append(
        {
            "split": main_split,
            "target": "",
            "regime": "high_volatility_week,spike_week",
            "severity": "info",
            "message": f"Selected high-volatility and spike weeks are restricted to the main reporting split: {main_split}.",
        }
    )
    selection_base = base.loc[base["split"].eq(main_split)].copy()
    hv_weeks = select_high_volatility_weeks(selection_base)
    spike_weeks = select_spike_weeks(selection_base, thresholds, epsilon=epsilon, warnings=warnings)
    selected_weeks = pd.concat([x for x in [hv_weeks, spike_weeks] if not x.empty], ignore_index=True) if (not hv_weeks.empty or not spike_weeks.empty) else pd.DataFrame()
    if not selected_weeks.empty:
        bad = selected_weeks.loc[~selected_weeks["split"].eq(main_split)]
        if not bad.empty:
            warnings.append({"split": main_split, "target": "", "regime": "selected_weeks", "severity": "error", "message": "Selected weeks outside the main reporting split were excluded."})
            selected_weeks = selected_weeks.loc[selected_weeks["split"].eq(main_split)].copy()

    metric_rows: list[dict[str, Any]] = []
    row_rows: list[dict[str, Any]] = []
    point_rows: list[pd.DataFrame] = []
    targets = sorted({target for _, _, target in loaded}, key=target_sort_key)
    for split in splits:
        for target in targets:
            if not all((m.key, split, target) in loaded for m in models):
                continue
            group, label = _target_info(target)
            base_target = loaded[(models[0].key, split, target)][["forecast_time_utc", "target_time_utc", "lead_time_h", "y_true", "target"]].copy()
            masks = regime_masks_for_target(base_target, thresholds, epsilon=epsilon)
            if not selected_weeks.empty:
                week = _week_start_utc(base_target["target_time_utc"])
                hv = selected_weeks[
                    (selected_weeks["split"].eq(split))
                    & (selected_weeks["target"].eq(target))
                    & (selected_weeks["selection_type"].eq("high_volatility_week"))
                ]
                sp = selected_weeks[
                    (selected_weeks["split"].eq(split))
                    & (selected_weeks["target"].eq(target))
                    & (selected_weeks["selection_type"].eq("spike_week"))
                ]
                if not hv.empty:
                    masks["high_volatility_week"] = week.isin(pd.to_datetime(hv["week_start_utc"], utc=True))
                if not sp.empty:
                    masks["spike_week"] = week.isin(pd.to_datetime(sp["week_start_utc"], utc=True))
            for regime, mask in masks.items():
                if int(mask.sum()) == 0:
                    warnings.append({"split": split, "target": target, "regime": regime, "severity": "warning", "message": "Regime selected zero rows."})
                    continue
                regime_keys = base_target.loc[mask, ["forecast_time_utc", "target_time_utc", "lead_time_h"]].copy()
                selected = {
                    m.key: loaded[(m.key, split, target)].merge(
                        regime_keys,
                        on=["forecast_time_utc", "target_time_utc", "lead_time_h"],
                        how="inner",
                    )
                    for m in models
                }
                common_qs = set.intersection(*(set(qmaps[(m.key, split, target)]) for m in models))
                if not common_qs:
                    raise ValueError(f"No common quantile grid for split={split}, target={target}, regime={regime}.")
                common_qcols = {m.key: {q: qmaps[(m.key, split, target)][q] for q in sorted(common_qs)} for m in models}
                valid = {m.key: _valid_frame(selected[m.key], common_qcols[m.key]) for m in models}
                common_keys = set.intersection(*(_key_tuples(valid[m.key]) for m in models))
                if not common_keys:
                    warnings.append({"split": split, "target": target, "regime": regime, "severity": "warning", "message": "Common valid row intersection is empty."})
                    continue
                key_df = pd.DataFrame(list(common_keys), columns=["forecast_time_utc", "target_time_utc", "lead_time_h"])
                quantiles_used = ",".join(_qcol(q) for q in sorted(common_qs))
                thr_row = thresholds[(thresholds["target"].eq(target)) & (thresholds["regime"].eq(regime))]
                threshold_value = float(thr_row["threshold_value"].iloc[0]) if not thr_row.empty else np.nan
                threshold_source = str(thr_row["threshold_source"].iloc[0]) if not thr_row.empty else ("weekly_selection" if regime in {"high_volatility_week", "spike_week"} else "")
                for model in models:
                    original_rows = int(len(selected[model.key]))
                    valid_rows = int(len(valid[model.key]))
                    retained_rows = int(len(common_keys))
                    row_rows.append(
                        {
                            "split": split,
                            "target": target,
                            "target_group": group,
                            "regime": regime,
                            "model": model.key,
                            "model_label": model.label,
                            "original_rows": original_rows,
                            "valid_rows": valid_rows,
                            "retained_common_rows": retained_rows,
                            "dropped_rows": int(valid_rows - retained_rows),
                            "retained_share": retained_rows / valid_rows if valid_rows else np.nan,
                            "quantiles_available": ",".join(_qcol(q) for q in sorted(qmaps[(model.key, split, target)])),
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
                            "target": target,
                            "target_label": label,
                            "target_group": group,
                            "regime": regime,
                            "quantiles_used": quantiles_used,
                            "regime_threshold_used": threshold_value,
                            "threshold_source": threshold_source,
                            **_metrics(eval_df, common_qcols[model.key]),
                        }
                    )
                    keep_cols = [
                        c
                        for c in [
                            "forecast_time_utc",
                            "target_time_utc",
                            "lead_time_h",
                            "y_true",
                            "p10",
                            "p30",
                            "p50",
                            "p70",
                            "p90",
                        ]
                        if c in eval_df.columns
                    ]
                    points = eval_df[keep_cols].copy()
                    points["model"] = model.key
                    points["model_label"] = model.label
                    points["split"] = split
                    points["target"] = target
                    points["target_label"] = label
                    points["target_group"] = group
                    points["regime"] = regime
                    point_rows.append(points)

    metrics_df = sort_target_frame(pd.DataFrame(metric_rows), target_col="target", extra_cols=["split", "regime", "model_label"])
    if not metrics_df.empty:
        available = metrics_df.loc[metrics_df["split"].eq(main_split)].groupby("target_label")["regime"].apply(lambda s: set(map(str, s))).to_dict()
        for target_label, regimes in MAIN_REGIMES_BY_TARGET_LABEL.items():
            have = available.get(target_label, set())
            for regime in regimes:
                if regime not in have:
                    warnings.append(
                        {
                            "split": main_split,
                            "target": target_label,
                            "regime": regime,
                            "severity": "warning",
                            "message": "Regime requested for thesis main tail/spike figure is unavailable for this target; no placeholder bar is drawn.",
                        }
                    )

    return {
        "metrics": metrics_df,
        "row_counts": sort_target_frame(pd.DataFrame(row_rows), target_col="target", extra_cols=["split", "regime", "model_label"]),
        "definitions": _regime_definitions(),
        "thresholds": thresholds,
        "selected_weeks": selected_weeks,
        "warnings": pd.DataFrame(warnings, columns=["split", "target", "regime", "severity", "message"]),
        "points": pd.concat(point_rows, ignore_index=True) if point_rows else pd.DataFrame(),
    }


def _latex_escape(value: Any) -> str:
    s = str(value)
    minus_token = "@@RQ1MINUS@@"
    for label in ["aFRR capacity price", "aFRR activation price", "aFRR activation rate", "capacity price", "activation price", "activation rate"]:
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


def _fmt_best(value: Any, *, model: str, best_model: Any) -> str:
    formatted = _fmt(value)
    if formatted != "-" and str(model) == str(best_model):
        return rf"\textbf{{{formatted}}}"
    return formatted


def _latex_header(label: str) -> str:
    return r"\textbf{" + _latex_escape(label) + r"}"


def write_latex_table(metrics: pd.DataFrame, *, out_dir: Path, split: str) -> Path | None:
    d = metrics.loc[metrics["split"].eq(split)].copy()
    if d.empty:
        return None
    rows: list[dict[str, Any]] = []
    d = sort_target_frame(d, target_col="target", extra_cols=["regime", "model_label"])
    for (target, target_label, regime), part in d.groupby(["target", "target_label", "regime"], sort=False):
        vals = {str(label): float(g["mean_pinball_loss"].mean()) for label, g in part.groupby("model_label")}
        best = min(vals, key=vals.get) if vals else ""
        rows.append(
            {
                "target": target,
                "target_label": str(target_label),
                "regime": regime,
                "regime_label": _clean_regime_label(regime),
                "RLQR": vals.get("RLQR", np.nan),
                "XGB": vals.get("XGB", np.nan),
                "TFT": vals.get("TFT", np.nan),
                "best_model": best,
                "n_obs": int(part.groupby("model_label")["n_obs"].sum().min()),
                "main_issue": _main_issue(regime),
            }
        )
    table = pd.DataFrame(rows)
    table["_target_order"] = table["target"].map(lambda x: target_sort_key(x)[0])
    table["_regime_order"] = table["regime"].map(lambda x: REGIME_ORDER.get(str(x), 99))
    table = table.sort_values(["_target_order", "_regime_order"]).drop(columns=["_target_order", "_regime_order"])

    def _compact_target_label(value: Any) -> str:
        label = str(value)
        if label.startswith("aFRR "):
            return label[len("aFRR ") :]
        return label

    headers = ["Regime", "Target", "RLQR", "XGB", "TFT"]
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \small",
        r"    \begin{tabular}{@{}llrrr@{}}",
        r"        \toprule",
        "        " + " & ".join(_latex_header(h) for h in headers) + r" \\",
        r"        \midrule",
    ]
    section = ""
    for _, row in table.iterrows():
        target_label = str(row["target_label"])
        next_section = "aFRR" if target_label.startswith("aFRR ") else ""
        if next_section and next_section != section:
            lines.append(r"        \multicolumn{5}{@{}l}{\textit{aFRR}} \\")
            section = next_section
        lines.append(
            "        "
            + " & ".join(
                [
                    _latex_escape(row["regime_label"]),
                    _latex_escape(_compact_target_label(target_label)),
                    _fmt_best(row["RLQR"], model="RLQR", best_model=row["best_model"]),
                    _fmt_best(row["XGB"], model="XGB", best_model=row["best_model"]),
                    _fmt_best(row["TFT"], model="TFT", best_model=row["best_model"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \caption{Tail and spike regime mean pinball loss on the test split. Metrics are unweighted and computed on common valid forecast rows across models.}",
            r"    \label{tab:tail_spike_metrics_test}",
            r"\end{table}",
            "",
        ]
    )
    path = out_dir / "latex" / f"tail_spike_metrics_{split}.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _plot_metric(metrics: pd.DataFrame, *, out_dir: Path, split: str, metric: str, filename: str, ylabel: str) -> Path | None:
    import matplotlib.pyplot as plt

    d = metrics.loc[metrics["split"].eq(split)].copy()
    if d.empty or metric not in d.columns:
        return None
    apply_geo_style()
    groups = sorted(d["target_group"].dropna().unique(), key=_target_group_sort_key)
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, max(3.0, 2.8 * len(groups))), sharex=False)
    if len(groups) == 1:
        axes = [axes]
    for ax, group in zip(axes, groups):
        p = d[d["target_group"].eq(group)].copy()
        agg = p.groupby(["regime", "model", "model_label"], as_index=False, sort=False).agg(value=(metric, "mean"), n_obs=("n_obs", "sum"))
        regimes = sorted(agg["regime"].unique(), key=lambda r: REGIME_ORDER.get(str(r), 99))
        x = np.arange(len(regimes), dtype=float)
        width = 0.24
        models = sorted(agg["model"].dropna().unique(), key=model_sort_key)
        for i, model in enumerate(models):
            mg = agg[agg["model"].eq(model)]
            vals = [float(mg.loc[mg["regime"].eq(r), "value"].mean()) for r in regimes]
            ax.bar(x + (i - 1) * width, vals, width=width, label=str(mg["model_label"].iloc[0]), color=get_model_color(str(model)))
        ax.set_title(thesis_titlecase(group))
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        labels = [_clean_regime_label(r) for r in regimes]
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.legend(ncol=3, loc="upper left")
    fig.suptitle(thesis_titlecase(f"{ylabel} by tail/spike regime ({split})"), y=1.01)
    fig.tight_layout()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def _relative_pinball_frame(metrics: pd.DataFrame, *, split: str) -> pd.DataFrame:
    d = metrics.loc[metrics["split"].eq(split)].copy()
    required = {"target_group", "target_label", "target", "regime", "model_label", "mean_pinball_loss", "n_obs"}
    if d.empty or not required <= set(d.columns):
        return pd.DataFrame()
    pivot = d.pivot_table(
        index=["target_group", "target_label", "target", "regime"],
        columns="model_label",
        values="mean_pinball_loss",
        aggfunc="mean",
    ).reset_index()
    if "RLQR" not in pivot.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    counts = d.groupby(["target_group", "target_label", "target", "regime", "model_label"], as_index=False)["n_obs"].min()
    n_by_key = counts.groupby(["target_group", "target_label", "target", "regime"], as_index=False)["n_obs"].min()
    pivot = pivot.merge(n_by_key, on=["target_group", "target_label", "target", "regime"], how="left")
    for label in ["XGB", "TFT"]:
        if label not in pivot.columns:
            continue
        part = pivot.loc[pivot["RLQR"].notna() & pivot[label].notna() & pivot["RLQR"].ne(0)].copy()
        part["model_label"] = label
        part["model"] = "tft" if label == "TFT" else "xgb"
        part["relative_pinball"] = part[label] / part["RLQR"]
        rows.extend(part[["target_group", "target_label", "target", "regime", "model", "model_label", "relative_pinball", "n_obs"]].to_dict("records"))
    return pd.DataFrame(rows)


def relative_pinball_plot_source(metrics: pd.DataFrame, *, split: str) -> pd.DataFrame:
    rel = _relative_pinball_frame(metrics, split=split)
    if rel.empty:
        return rel
    d = metrics.loc[metrics["split"].eq(split)].copy()
    rlqr = d.loc[d["model_label"].eq("RLQR"), ["target_group", "target_label", "target", "regime", "mean_pinball_loss"]].rename(
        columns={"mean_pinball_loss": "rlqr_mean_pinball_loss"}
    )
    model_vals = d[["target_group", "target_label", "target", "regime", "model", "model_label", "mean_pinball_loss", "n_obs"]].copy()
    out = model_vals.merge(rlqr, on=["target_group", "target_label", "target", "regime"], how="inner")
    out = out.loc[out["model_label"].isin(["XGB", "TFT"])].copy()
    out["relative_pinball_loss"] = out["mean_pinball_loss"] / out["rlqr_mean_pinball_loss"].replace(0.0, np.nan)
    out = out.rename(columns={"target_label": "target_display"})
    cols = [
        "split",
        "target",
        "target_display",
        "target_group",
        "regime",
        "model",
        "model_label",
        "mean_pinball_loss",
        "rlqr_mean_pinball_loss",
        "relative_pinball_loss",
        "n_obs",
    ]
    out["split"] = split
    return out[[c for c in cols if c in out.columns]]


def residual_distribution_source(points: pd.DataFrame, *, split: str) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame()
    d = points.loc[points["split"].eq(split)].copy()
    if d.empty:
        return d
    d = d.rename(columns={"target_label": "target_display"})
    d["residual_p50"] = pd.to_numeric(d["p50"], errors="coerce") - pd.to_numeric(d["y_true"], errors="coerce")
    d["abs_error_p50"] = d["residual_p50"].abs()
    cols = [
        "split",
        "target",
        "target_display",
        "target_group",
        "regime",
        "model",
        "model_label",
        "target_time_utc",
        "forecast_time_utc",
        "lead_time_h",
        "y_true",
        "p50",
        "residual_p50",
        "abs_error_p50",
    ]
    return d[[c for c in cols if c in d.columns]].copy()


def _plot_relative_pinball_all_in_one(metrics: pd.DataFrame, *, out_dir: Path, split: str) -> Path | None:
    import matplotlib.pyplot as plt

    rel = _relative_pinball_frame(metrics, split=split)
    if rel.empty:
        return None

    apply_geo_style()
    groups = sorted(rel["target_group"].dropna().unique(), key=_target_group_sort_key)
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, max(3.4, 3.0 * len(groups))), sharex=False)
    if len(groups) == 1:
        axes = [axes]
    for ax, group in zip(axes, groups):
        p = rel[rel["target_group"].eq(group)].copy()
        agg = p.groupby(["regime", "model", "model_label"], as_index=False, sort=False).agg(
            value=("relative_pinball", "mean"),
            n_obs=("n_obs", "min"),
        )
        regimes = sorted(agg["regime"].unique(), key=lambda r: REGIME_ORDER.get(str(r), 99))
        x = np.arange(len(regimes), dtype=float)
        width = 0.32
        ax.axhline(1.0, color=THESIS_PALETTE["neutral_dark"], linestyle="--", linewidth=1.5, label="RLQR baseline")
        for i, label in enumerate(["XGB", "TFT"]):
            mg = agg[agg["model_label"].eq(label)]
            if mg.empty:
                continue
            vals = [float(mg.loc[mg["regime"].eq(r), "value"].mean()) if mg["regime"].eq(r).any() else np.nan for r in regimes]
            model_key = "tft" if label == "TFT" else "xgb"
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=label, color=get_model_color(model_key))
        n_labels = []
        for regime in regimes:
            n = agg.loc[agg["regime"].eq(regime), "n_obs"].min()
            n_labels.append(f"{_clean_regime_label(regime)}\nN={int(n)}" if pd.notna(n) else _clean_regime_label(regime))
        ax.set_title(thesis_titlecase(str(group)))
        ax.set_ylabel("Mean pinball / RLQR")
        ax.set_xticks(x)
        ax.set_xticklabels(n_labels, rotation=25, ha="right")
        ax.legend(ncol=3, loc="upper left")
    fig.suptitle(thesis_titlecase(f"Tail/spike mean pinball loss relative to RLQR ({split})"), y=1.01)
    fig.tight_layout()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "tail_spike_relative_pinball_by_regime_all_in_one.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_relative_pinball_main_figure(
    rel: pd.DataFrame,
    *,
    out_dir: Path,
    target_labels: list[str],
    filename: str,
    title: str,
) -> Path | None:
    import matplotlib.pyplot as plt

    rows: list[pd.DataFrame] = []
    for target_label in target_labels:
        regimes = MAIN_REGIMES_BY_TARGET_LABEL[target_label]
        part = rel[rel["target_label"].eq(target_label) & rel["regime"].isin(regimes)].copy()
        if part.empty:
            continue
        part["_target_order"] = target_labels.index(target_label)
        part["_regime_order"] = part["regime"].map(MAIN_REGIME_ORDER[target_label]).fillna(99)
        rows.append(part)
    if not rows:
        return None
    main = pd.concat(rows, ignore_index=True)

    apply_geo_style()
    labels = [target_label for target_label in target_labels if target_label in set(main["target_label"])]
    fig_height = max(4.4, 2.45 * len(labels))
    fig, axes = plt.subplots(len(labels), 1, figsize=(11.4, fig_height), sharex=True)
    axes_flat = [axes] if len(labels) == 1 else list(axes)
    for ax, target_label in zip(axes_flat, labels):
        p = main[main["target_label"].eq(target_label)].copy()
        agg = (
            p.groupby(["regime", "model", "model_label"], as_index=False, sort=False)
            .agg(value=("relative_pinball", "mean"), n_obs=("n_obs", "min"))
        )
        regimes = sorted(agg["regime"].dropna().unique(), key=lambda r: _main_regime_order_key(target_label, r))
        y = np.arange(len(regimes), dtype=float)
        height = 0.42
        center_sep = height * 0.5
        ax.axvline(1.0, color=THESIS_PALETTE["neutral_dark"], linestyle="--", linewidth=1.3, label="RLQR baseline")
        for i, label in enumerate(["XGB", "TFT"]):
            mg = agg[agg["model_label"].eq(label)]
            if mg.empty:
                continue
            vals = [float(mg.loc[mg["regime"].eq(r), "value"].mean()) if mg["regime"].eq(r).any() else np.nan for r in regimes]
            model_key = "tft" if label == "TFT" else "xgb"
            ax.barh(y + (i - 0.5) * center_sep, vals, height=height, label=label, color=get_model_color(model_key), alpha=1.0)
        ax.set_title(thesis_titlecase(target_label))
        ax.set_yticks(y)
        ax.set_yticklabels([_clean_regime_label(r) for r in regimes])
        ax.set_ylim(len(regimes) - 0.45, -0.45)
        ax.grid(axis="x", alpha=0.25)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False)
    axes_flat[-1].set_xlabel("Mean pinball loss relative to RLQR")
    fig.suptitle(thesis_titlecase(title), y=0.965)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_relative_pinball_main(metrics: pd.DataFrame, *, out_dir: Path, split: str) -> list[Path]:
    rel = _relative_pinball_frame(metrics, split=split)
    if rel.empty:
        return []
    paths: list[Path] = []
    for target_labels, filename, title in [
        (
            MAIN_PRICE_CAPACITY_TARGETS,
            "tail_spike_relative_pinball_by_regime_price_capacity.png",
            "Tail and Spike Performance by Regime: DA and aFRR Capacity",
        ),
        (
            MAIN_ACTIVATION_TARGETS,
            "tail_spike_relative_pinball_by_regime_activation.png",
            "Tail and Spike Performance by Regime: aFRR Activation",
        ),
    ]:
        path = _plot_relative_pinball_main_figure(rel, out_dir=out_dir, target_labels=target_labels, filename=filename, title=title)
        if path is not None:
            paths.append(path)
    return paths


def _plot_hexbin(metrics_source: pd.DataFrame, *, out_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    if metrics_source.empty:
        return None
    apply_geo_style()
    d = metrics_source.copy()
    groups = sorted(d["target_group"].dropna().unique(), key=_target_group_sort_key)
    models = [m for m in ordered_model_labels(d["model_label"].dropna().unique()) if m in set(d["model_label"])]
    if not groups or not models:
        return None
    fig, axes = plt.subplots(len(groups), len(models), figsize=(4.2 * len(models), 3.2 * len(groups)), squeeze=False)
    for r, group in enumerate(groups):
        for c, model_label in enumerate(models):
            ax = axes[r][c]
            p = d[d["target_group"].eq(group) & d["model_label"].eq(model_label)]
            if p.empty:
                ax.axis("off")
                continue
            ax.hexbin(p["y_true"], p["p50"], gridsize=28, mincnt=1, cmap="Blues")
            lo = float(np.nanmin([p["y_true"].min(), p["p50"].min()]))
            hi = float(np.nanmax([p["y_true"].max(), p["p50"].max()]))
            ax.plot([lo, hi], [lo, hi], linestyle="--", color=THESIS_PALETTE["neutral_dark"], linewidth=1.2)
            ax.set_title(thesis_titlecase(f"{group}: {model_label}"))
            ax.set_xlabel("Truth")
            ax.set_ylabel("p50 forecast")
    fig.suptitle(thesis_titlecase("Truth vs p50 forecast density by target group"), y=1.01)
    fig.tight_layout()
    path = out_dir / "figures" / "tail_spike_true_vs_p50_hexbin.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_residual_distribution(metrics_source: pd.DataFrame, *, out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    if metrics_source.empty:
        return []
    apply_geo_style()
    d = metrics_source.copy()
    d["residual"] = d["p50"] - d["y_true"]
    d["regime_class"] = np.where(d["regime"].eq("normal"), "Normal", "Tail/spike")
    d = d.drop_duplicates(
        subset=["model", "target_group", "target", "regime_class", "forecast_time_utc", "target_time_utc", "lead_time_h"]
    )
    groups = sorted(d["target_group"].dropna().unique(), key=_target_group_sort_key)
    models = [m for m in ordered_model_labels(d["model_label"].dropna().unique()) if m in set(d["model_label"])]
    if not groups or not models:
        return []
    fig, axes = plt.subplots(len(groups), len(models), figsize=(4.2 * len(models), 3.0 * len(groups)), squeeze=False)
    colors = {"Normal": THESIS_PALETTE["naive"], "Tail/spike": THESIS_PALETTE["primary"]}
    for r, group in enumerate(groups):
        for c, model_label in enumerate(models):
            ax = axes[r][c]
            p = d[d["target_group"].eq(group) & d["model_label"].eq(model_label)]
            if p.empty:
                ax.axis("off")
                continue
            for regime_class in ["Normal", "Tail/spike"]:
                g = p[p["regime_class"].eq(regime_class)]
                if g.empty:
                    continue
                ax.hist(
                    g["residual"],
                    bins=35,
                    alpha=0.38,
                    label=regime_class,
                    density=True,
                    color=colors[regime_class],
                )
            ax.axvline(0.0, color=THESIS_PALETTE["neutral_dark"], linestyle="--", linewidth=1.1)
            ax.set_title(thesis_titlecase(f"{group}: {model_label}"))
            ax.set_xlabel("p50 prediction - realized value")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
    fig.suptitle(thesis_titlecase("Residual distribution in normal vs tail/spike regimes"), y=1.01)
    fig.tight_layout()
    path = out_dir / "figures" / "tail_spike_residual_distribution_by_regime.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    paths = [path]
    for group in groups:
        p = d[d["target_group"].eq(group)].copy()
        if p.empty:
            continue
        fig_g, axes_g = plt.subplots(len(models), 1, figsize=(8, max(3.0, 2.7 * len(models))), squeeze=False)
        for i, model_label in enumerate(models):
            ax = axes_g[i][0]
            m = p[p["model_label"].eq(model_label)]
            if m.empty:
                ax.axis("off")
                continue
            for regime_class in ["Normal", "Tail/spike"]:
                g = m[m["regime_class"].eq(regime_class)]
                if g.empty:
                    continue
                ax.hist(g["residual"], bins=35, alpha=0.38, label=regime_class, density=True, color=colors[regime_class])
            ax.axvline(0.0, color=THESIS_PALETTE["neutral_dark"], linestyle="--", linewidth=1.1)
            ax.set_title(thesis_titlecase(model_label))
            ax.set_xlabel("p50 prediction - realized value")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
        fig_g.suptitle(thesis_titlecase(f"Residual distribution: {group}"), y=1.01)
        fig_g.tight_layout()
        group_path = out_dir / "figures" / f"tail_spike_residual_distribution_{_target_group_slug(group)}.png"
        fig_g.savefig(group_path)
        plt.close(fig_g)
        paths.append(group_path)
    return paths


def plot_forecast_band_example(df: pd.DataFrame, *, out_path: Path, lead: int | None = None, snapshot: str | None = None) -> Path:
    if (lead is None) == (snapshot is None):
        raise ValueError("Forecast-band plot requires exactly one lead or one actionable snapshot.")
    d = df.copy()
    if lead is not None:
        d = d.loc[pd.to_numeric(d["lead_time_h"], errors="coerce").eq(float(lead))].copy()
        selector = f"lead_h{int(lead)}"
    else:
        snap = pd.to_datetime(snapshot, utc=True)
        d = d.loc[pd.to_datetime(d["forecast_time_utc"], utc=True).eq(snap)].copy()
        selector = "snapshot_" + snap.strftime("%Y%m%dT%H%MZ")
    if d.empty:
        raise ValueError(f"No rows selected for forecast-band example {selector}.")
    import matplotlib.pyplot as plt

    apply_geo_style()
    d = d.sort_values("target_time_utc")
    fig, ax = plt.subplots(figsize=(10, 4))
    x = pd.to_datetime(d["target_time_utc"], utc=True)
    ax.plot(x, d["y_true"], color=THESIS_PALETTE["perfect_foresight"], label="Truth", linewidth=2.0)
    ax.plot(x, d["p50"], color=THESIS_PALETTE["primary"], label="p50", linewidth=1.8)
    if {"p10", "p90"} <= set(d.columns):
        ax.fill_between(x, d["p10"], d["p90"], color=THESIS_PALETTE["secondary"], alpha=0.22, label="p10-p90")
    if {"p30", "p70"} <= set(d.columns):
        ax.fill_between(x, d["p30"], d["p70"], color=THESIS_PALETTE["tertiary"], alpha=0.18, label="p30-p70")
    ax.set_title(thesis_titlecase(f"Forecast-band example ({selector})"))
    ax.set_xlabel("Target time")
    ax.set_ylabel("Value")
    ax.legend(ncol=4, loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def select_example_week(
    selected_weeks: pd.DataFrame,
    *,
    split: str,
    selection_type: str | None = None,
    target: str | None = None,
    target_group: str | None = None,
) -> tuple[pd.Series | None, str, str, float]:
    if selected_weeks.empty:
        return None, "", "", float("nan")
    weeks = selected_weeks.loc[selected_weeks["split"].eq(split)].copy()
    if weeks.empty:
        return None, "", "", float("nan")
    if selection_type is not None:
        weeks = weeks.loc[weeks["selection_type"].eq(selection_type)].copy()
    if target is not None and "target" in weeks.columns:
        weeks = weeks.loc[weeks["target"].eq(canonical_target(target))].copy()
    if target_group is not None and "target_group" in weeks.columns:
        weeks = weeks.loc[weeks["target_group"].eq(target_group)].copy()
    if weeks.empty:
        return None, "", "", float("nan")

    stype = str(selection_type or weeks["selection_type"].iloc[0])
    if stype == "high_volatility_week":
        ordered = weeks.sort_values(["volatility_score", "n_rows", "week_start_utc"], ascending=[False, False, True]).copy()
        metric = "volatility_score"
        reason_metric = "highest volatility_score"
    elif stype == "spike_week":
        ordered = weeks.sort_values(["spike_share", "spike_count", "n_rows", "week_start_utc"], ascending=[False, False, False, True]).copy()
        metric = "spike_share"
        reason_metric = "highest spike_share and spike_count"
    else:
        ordered = sort_target_frame(weeks, target_col="target", extra_cols=["selection_type", "week_start_utc"]).copy()
        metric = "week_start_utc"
        reason_metric = "deterministic selected-week order"
    ordered["_example_selection_rank"] = np.arange(1, len(ordered) + 1)
    row = ordered.iloc[0]
    metric_value = float(row[metric]) if metric in row.index and pd.notna(row[metric]) and metric != "week_start_utc" else float("nan")
    target_label = str(row.get("target_display", row.get("target", "")))
    reason = f"Selected {reason_metric} among {stype} rows for split={split} and target={target_label}."
    return row, metric, reason, metric_value


def _plot_selected_week_band(
    points: pd.DataFrame,
    selected_weeks: pd.DataFrame,
    *,
    out_dir: Path,
    split: str,
    selection_type: str | None = None,
    target: str | None = None,
    target_group: str | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
    plot_values: list[pd.DataFrame] | None = None,
    warnings: list[dict[str, Any]] | None = None,
) -> Path | None:
    if points.empty or selected_weeks.empty:
        return None
    week, selection_metric, selection_reason, selection_metric_value = select_example_week(
        selected_weeks,
        split=split,
        selection_type=selection_type,
        target=target,
        target_group=target_group,
    )
    if week is None:
        return None
    start = pd.to_datetime(week["week_start_utc"], utc=True)
    end = start + pd.Timedelta(days=7)
    selected_target = canonical_target(str(week["target"])) if "target" in week.index else ""
    d = points.loc[
        points["split"].eq(split)
        & points["target"].eq(selected_target)
        & pd.to_datetime(points["target_time_utc"], utc=True).between(start, end, inclusive="left")
    ].copy()
    if d.empty:
        return None
    model_order = {"xgb": 0, "tft": 1, "linear": 2}
    d["_model_order"] = d["model"].map(model_order).fillna(99)
    model = str(d.sort_values("_model_order")["model"].iloc[0])
    d = d[d["model"].eq(model)].copy()
    lead = int(pd.to_numeric(d["lead_time_h"], errors="coerce").dropna().min())
    plot_d = d.loc[pd.to_numeric(d["lead_time_h"], errors="coerce").eq(float(lead))].copy()
    group_slug = str(week["target_display"] if "target_display" in week.index else week["target_group"]).lower().replace(" ", "_").replace("+", "pos").replace("-", "neg").replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    selection = str(week["selection_type"])
    out_path = out_dir / "figures" / f"tail_spike_forecast_band_{selection}_{group_slug}_lead_h{lead}.png"
    if diagnostics is not None:
        diagnostics.append(
            {
                "split": split,
                "selection_type": selection,
                "target": selected_target,
                "target_display": str(week.get("target_display", selected_target)),
                "target_group": str(week.get("target_group", "")),
                "week_start_utc": start.isoformat(),
                "week_end_utc": end.isoformat(),
                "model": model,
                "model_label": str(d["model_label"].iloc[0]) if "model_label" in d.columns and not d.empty else model,
                "lead_time_h": lead,
                "selection_metric": selection_metric,
                "selection_metric_value": selection_metric_value,
                "selection_rank": int(week.get("_example_selection_rank", 1)),
                "selection_reason": selection_reason,
                "n_rows": int(week.get("n_rows", len(d))),
                "source_selected_weeks_file": "tail_spike_selected_weeks.csv",
            }
        )
    if plot_values is not None:
        value_df = plot_d.copy()
        for col in ["p10", "p90"]:
            if col not in value_df.columns:
                value_df[col] = np.nan
                if warnings is not None:
                    warnings.append(
                        {
                            "split": split,
                            "target": selected_target,
                            "regime": selection,
                            "severity": "warning",
                            "message": f"Example-week plot source is missing {col}; column is written as missing in tail_spike_example_week_plot_values.csv.",
                        }
                    )
        value_df = value_df.rename(columns={"target_label": "target_display"})
        value_df["selection_type"] = selection
        value_df["week_start_utc"] = start.isoformat()
        value_df["week_end_utc"] = end.isoformat()
        value_df["selection_metric"] = selection_metric
        value_df["selection_metric_value"] = selection_metric_value
        value_df["selection_reason"] = selection_reason
        value_df["residual_p50"] = pd.to_numeric(value_df["p50"], errors="coerce") - pd.to_numeric(value_df["y_true"], errors="coerce")
        cols = [
            "selection_type",
            "split",
            "target",
            "target_display",
            "target_group",
            "week_start_utc",
            "week_end_utc",
            "model",
            "model_label",
            "lead_time_h",
            "target_time_utc",
            "forecast_time_utc",
            "y_true",
            "p10",
            "p50",
            "p90",
            "residual_p50",
            "selection_metric",
            "selection_metric_value",
            "selection_reason",
        ]
        plot_values.append(value_df[[c for c in cols if c in value_df.columns]].copy())
    return plot_forecast_band_example(d, out_path=out_path, lead=lead)


def validate_tail_spike_outputs(
    *,
    out_dir: Path,
    split: str,
    selected_weeks: pd.DataFrame,
    points_test: pd.DataFrame,
    relative_source: pd.DataFrame,
    example_selection: pd.DataFrame,
    example_values: pd.DataFrame,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if not selected_weeks.empty and not selected_weeks["split"].eq(split).all():
        raise ValueError("tail_spike_selected_weeks contains rows outside the main reporting split.")
    if points_test.empty:
        warnings.append({"split": split, "target": "", "regime": "points", "severity": "warning", "message": "tail_spike_points_test.csv source data is empty."})
    if relative_source.empty:
        warnings.append({"split": split, "target": "", "regime": "relative_pinball", "severity": "warning", "message": "Relative pinball plot source data is empty for the plotted split."})
    if not example_selection.empty and example_values.empty:
        warnings.append({"split": split, "target": "", "regime": "example_week", "severity": "warning", "message": "Example-week selection exists but tail_spike_example_week_plot_values.csv is empty."})
    if not example_values.empty and example_selection.empty:
        warnings.append({"split": split, "target": "", "regime": "example_week", "severity": "warning", "message": "Example-week plot values exist but tail_spike_example_week_selection.csv is empty."})
    for filename, frame in [
        ("tail_spike_points_test.csv", points_test),
        ("tail_spike_plot_source_relative_pinball.csv", relative_source),
        ("tail_spike_example_week_plot_values.csv", example_values),
    ]:
        path = out_dir / filename
        if not path.exists():
            warnings.append({"split": split, "target": "", "regime": "output_validation", "severity": "warning", "message": f"Expected plot/source CSV was not written: {filename}."})
        elif frame.empty:
            warnings.append({"split": split, "target": "", "regime": "output_validation", "severity": "warning", "message": f"Plot/source CSV is empty: {filename}."})
    return warnings


def write_outputs(
    outputs: dict[str, pd.DataFrame],
    *,
    out_dir: Path,
    split: str,
    structured_out_dir: Path | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    runtime_warnings = outputs.get("warnings", pd.DataFrame()).copy()
    points = outputs.get("points", pd.DataFrame())
    points_split = points.loc[points["split"].eq(split)].copy() if not points.empty and "split" in points.columns else pd.DataFrame()
    rel_source = relative_pinball_plot_source(outputs["metrics"], split=split)
    rel_price_capacity = rel_source.loc[rel_source["target_display"].isin(MAIN_PRICE_CAPACITY_TARGETS)].copy() if not rel_source.empty else pd.DataFrame()
    rel_activation = rel_source.loc[rel_source["target_display"].isin(MAIN_ACTIVATION_TARGETS)].copy() if not rel_source.empty else pd.DataFrame()
    residual_source = residual_distribution_source(points, split=split)
    hexbin_source = points_split.copy()
    if not hexbin_source.empty:
        hexbin_source = hexbin_source.rename(columns={"target_label": "target_display"})
    if rel_source.empty:
        runtime_warnings = pd.concat(
            [runtime_warnings, pd.DataFrame([{"split": split, "target": "", "regime": "relative_pinball", "severity": "warning", "message": "Relative pinball plot source data is empty for the plotted split."}])],
            ignore_index=True,
        )
    if residual_source.empty:
        runtime_warnings = pd.concat(
            [runtime_warnings, pd.DataFrame([{"split": split, "target": "", "regime": "residual_distribution", "severity": "warning", "message": "Residual distribution plot source data is empty for the plotted split."}])],
            ignore_index=True,
        )
    csvs = [
        ("tail_spike_metrics.csv", outputs["metrics"]),
        (f"tail_spike_metrics_{split}.csv", outputs["metrics"].loc[outputs["metrics"]["split"].eq(split)]),
        ("tail_spike_regime_definitions.csv", outputs["definitions"]),
        ("tail_spike_thresholds.csv", outputs["thresholds"]),
        ("tail_spike_row_counts.csv", outputs["row_counts"]),
        ("tail_spike_selected_weeks.csv", outputs["selected_weeks"]),
        ("tail_spike_warnings.csv", runtime_warnings),
        ("tail_spike_selection_warnings.csv", runtime_warnings),
        ("tail_spike_points.csv", points),
        (f"tail_spike_points_{split}.csv", points_split),
        ("tail_spike_plot_source_relative_pinball.csv", rel_source),
        ("tail_spike_plot_source_relative_pinball_price_capacity.csv", rel_price_capacity),
        ("tail_spike_plot_source_relative_pinball_activation.csv", rel_activation),
        ("tail_spike_plot_source_residual_distribution.csv", residual_source),
        ("tail_spike_residual_distribution.csv", residual_source),
        ("tail_spike_plot_source_hexbin.csv", hexbin_source),
    ]
    for name, df in csvs:
        path = out_dir / name
        df.to_csv(path, index=False)
        paths.append(path)
    tex = write_latex_table(outputs["metrics"], out_dir=out_dir, split=split)
    if tex is not None:
        paths.append(tex)
    example_week_selection: list[dict[str, Any]] = []
    example_week_values: list[pd.DataFrame] = []
    plot_warnings: list[dict[str, Any]] = []
    plot_results = [
        _plot_relative_pinball_main(outputs["metrics"], out_dir=out_dir, split=split),
        _plot_relative_pinball_all_in_one(outputs["metrics"], out_dir=out_dir, split=split),
        _plot_metric(outputs["metrics"], out_dir=out_dir, split=split, metric="mean_pinball_loss", filename="tail_spike_pinball_by_regime.png", ylabel="Mean pinball loss"),
        _plot_metric(outputs["metrics"], out_dir=out_dir, split=split, metric="mae_p50", filename="tail_spike_mae_p50_by_regime.png", ylabel="MAE p50"),
        _plot_metric(outputs["metrics"], out_dir=out_dir, split=split, metric="coverage_p10_p90", filename="tail_spike_coverage_by_regime.png", ylabel="p10-p90 coverage"),
        _plot_hexbin(outputs.get("points", pd.DataFrame()).loc[outputs.get("points", pd.DataFrame()).get("split", pd.Series(dtype=str)).eq(split)] if not outputs.get("points", pd.DataFrame()).empty else pd.DataFrame(), out_dir=out_dir),
        _plot_residual_distribution(outputs.get("points", pd.DataFrame()).loc[outputs.get("points", pd.DataFrame()).get("split", pd.Series(dtype=str)).eq(split)] if not outputs.get("points", pd.DataFrame()).empty else pd.DataFrame(), out_dir=out_dir),
        _plot_selected_week_band(outputs.get("points", pd.DataFrame()), outputs.get("selected_weeks", pd.DataFrame()), out_dir=out_dir, split=split, selection_type="high_volatility_week", diagnostics=example_week_selection, plot_values=example_week_values, warnings=plot_warnings),
        _plot_selected_week_band(outputs.get("points", pd.DataFrame()), outputs.get("selected_weeks", pd.DataFrame()), out_dir=out_dir, split=split, selection_type="spike_week", diagnostics=example_week_selection, plot_values=example_week_values, warnings=plot_warnings),
    ]
    for p in plot_results:
        if p is not None:
            if isinstance(p, list):
                paths.extend(p)
            else:
                paths.append(p)
    example_path = out_dir / "tail_spike_example_week_selection.csv"
    example_selection_df = pd.DataFrame(example_week_selection)
    example_selection_df.to_csv(example_path, index=False)
    paths.append(example_path)
    example_values_path = out_dir / "tail_spike_example_week_plot_values.csv"
    example_values_df = pd.concat(example_week_values, ignore_index=True) if example_week_values else pd.DataFrame()
    example_values_df.to_csv(example_values_path, index=False)
    paths.append(example_values_path)
    validation_warnings = validate_tail_spike_outputs(
        out_dir=out_dir,
        split=split,
        selected_weeks=outputs["selected_weeks"],
        points_test=points_split,
        relative_source=rel_source,
        example_selection=example_selection_df,
        example_values=example_values_df,
    )
    if validation_warnings:
        plot_warnings.extend(validation_warnings)
    if plot_warnings:
        runtime_warnings = pd.concat([runtime_warnings, pd.DataFrame(plot_warnings)], ignore_index=True)
        for warning_name in ["tail_spike_warnings.csv", "tail_spike_selection_warnings.csv"]:
            runtime_warnings.to_csv(out_dir / warning_name, index=False)
    manifest = {
        "description": "RQ1 tail/spike behavior outputs.",
        "split": split,
        "regimes": outputs["definitions"]["regime"].tolist(),
        "row_intersection_key": "split,target,forecast_time_utc,target_time_utc,lead_time_h",
        "outputs": [str(p) for p in paths],
    }
    manifest_path = out_dir / "tail_spike_manifest.json"
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
        if dst.exists() or dst.is_symlink():
            try:
                if src.samefile(dst):
                    mirrored.append(dst)
                    continue
            except OSError:
                pass
            dst.unlink()
        try:
            os.link(src, dst)
            shutil.copystat(src, dst, follow_symlinks=True)
        except OSError:
            shutil.copy2(src, dst)
        mirrored.append(dst)
    return mirrored


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build final RQ1 tail/spike benchmark outputs.")
    p.add_argument("--benchmark-root", default="artifacts")
    p.add_argument("--benchmark-dir", default=None)
    p.add_argument("--out-dir", default="artifacts/benchmark/rq1_ml_model_benchmark/_raw_outputs/shared")
    p.add_argument("--structured-out-dir", default="artifacts/benchmark/rq1_ml_model_benchmark/_raw_outputs/4_1_5_tail_spike")
    p.add_argument("--split", default="test", help="Main thesis/reporting split. Defaults to test.")
    p.add_argument(
        "--splits",
        default="test",
        help=(
            "Comma-separated splits to load/export. Defaults to test only; tail/spike thresholds are therefore test-conditional."
        ),
    )
    p.add_argument("--models", default="tft,xgboost,linear", help="Models to compare. Defaults to all RQ1 models: TFT, XGB and RLQR.")
    p.add_argument("--epsilon", type=float, default=EPSILON_DEFAULT)
    p.add_argument("--threshold-source", choices=["test", "validation", "train_validation"], default="test", help="Source split(s) for tail/spike thresholds. Defaults to test for thesis outputs.")
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
    benchmark_dir = _discover_benchmark_dir(Path(args.benchmark_root), Path(args.benchmark_dir) if args.benchmark_dir else None)
    outputs = build_tail_spike_outputs(
        benchmark_dir=benchmark_dir,
        models=models,
        splits=splits,
        main_split=args.split,
        epsilon=float(args.epsilon),
        threshold_source=str(args.threshold_source),
        derive_forecast_time=bool(args.derive_forecast_time_from_lead),
        eval_origin_start=_parse_utc_bound(args.eval_origin_start),
        eval_origin_end=_parse_utc_bound(args.eval_origin_end),
    )
    structured_out_dir = None if args.no_structured_copy else Path(args.structured_out_dir)
    paths = write_outputs(outputs, out_dir=Path(args.out_dir), split=args.split, structured_out_dir=structured_out_dir)
    print("[OK] Built RQ1 tail/spike outputs.")
    print(f"[OK] benchmark_dir={benchmark_dir}")
    for path in paths:
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
