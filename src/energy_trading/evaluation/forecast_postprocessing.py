from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd


MANDATORY_QUANTILES = ["p10", "p30", "p50", "p70", "p90"]
OPTIONAL_SYMMETRIC_PAIRS = [("p01", "p99"), ("p05", "p95")]
MANDATORY_SYMMETRIC_PAIRS = [("p10", "p90"), ("p30", "p70"), ("p50", "p50")]


@dataclass(frozen=True)
class TargetPostprocessingSpec:
    target_name: str
    semantic_type: str
    sign_transform: str
    quantile_flip: bool
    clip_lower: float | None = None
    clip_upper: float | None = None


SPECS: dict[str, TargetPostprocessingSpec] = {
    "pred_da_price": TargetPostprocessingSpec(
        target_name="pred_da_price",
        semantic_type="market_price",
        sign_transform="none",
        quantile_flip=False,
    ),
    "pred_afrr_capacity_price_pos": TargetPostprocessingSpec(
        target_name="pred_afrr_capacity_price_pos",
        semantic_type="provider_value",
        sign_transform="none",
        quantile_flip=False,
        clip_lower=0.0,
    ),
    "pred_afrr_capacity_price_neg": TargetPostprocessingSpec(
        target_name="pred_afrr_capacity_price_neg",
        semantic_type="market_price_or_value_neg",
        sign_transform="none",
        quantile_flip=False,
    ),
    "pred_afrr_activation_price_pos": TargetPostprocessingSpec(
        target_name="pred_afrr_activation_price_pos",
        semantic_type="market_or_provider_value_pos",
        sign_transform="none",
        quantile_flip=False,
    ),
    "pred_afrr_activation_price_neg": TargetPostprocessingSpec(
        target_name="pred_afrr_activation_price_neg",
        semantic_type="provider_value_neg_mode_aware",
        sign_transform="none",
        quantile_flip=False,
        clip_lower=0.0,
    ),
    "pred_afrr_activation_rate_pos": TargetPostprocessingSpec(
        target_name="pred_afrr_activation_rate_pos",
        semantic_type="rate",
        sign_transform="none",
        quantile_flip=False,
        clip_lower=0.0,
        clip_upper=1.0,
    ),
    "pred_afrr_activation_rate_neg": TargetPostprocessingSpec(
        target_name="pred_afrr_activation_rate_neg",
        semantic_type="rate",
        sign_transform="none",
        quantile_flip=False,
        clip_lower=0.0,
        clip_upper=1.0,
    ),
}

TRUTH_TO_PRED_TARGET: dict[str, str] = {
    "da_price": "pred_da_price",
    "target_da_price": "pred_da_price",
    "afrr_capacity_price_pos": "pred_afrr_capacity_price_pos",
    "target_afrr_capacity_price_pos": "pred_afrr_capacity_price_pos",
    "afrr_capacity_price_neg": "pred_afrr_capacity_price_neg",
    "target_afrr_capacity_price_neg": "pred_afrr_capacity_price_neg",
    "afrr_activation_price_vwap_pos": "pred_afrr_activation_price_pos",
    "target_afrr_activation_price_vwap_pos": "pred_afrr_activation_price_pos",
    "afrr_activation_price_vwap_neg": "pred_afrr_activation_price_neg",
    "target_afrr_activation_price_vwap_neg": "pred_afrr_activation_price_neg",
    "activation_rate_phys_pos": "pred_afrr_activation_rate_pos",
    "target_afrr_activation_rate_pos": "pred_afrr_activation_rate_pos",
    "activation_rate_phys_neg": "pred_afrr_activation_rate_neg",
    "target_afrr_activation_rate_neg": "pred_afrr_activation_rate_neg",
}


def get_target_postprocessing_spec(target_name: str, *, target_value_mode: str | None = None) -> TargetPostprocessingSpec:
    if target_name not in SPECS:
        raise KeyError(f"No postprocessing spec for target: {target_name}")
    spec = SPECS[target_name]
    if target_name == "pred_afrr_activation_price_neg":
        mode = str(target_value_mode or "raw_signed_legacy").strip().lower()
        if mode == "raw_signed_legacy":
            return replace(
                spec,
                semantic_type="provider_value_from_raw_negative",
                sign_transform="multiply_by_minus_one",
                quantile_flip=True,
                clip_lower=0.0,
            )
        return replace(
            spec,
            semantic_type="provider_value_canonical_economic",
            sign_transform="none",
            quantile_flip=False,
            clip_lower=0.0,
        )
    return spec


def target_requires_quantile_flip(target_name: str, *, target_value_mode: str | None = None) -> bool:
    return bool(get_target_postprocessing_spec(target_name, target_value_mode=target_value_mode).quantile_flip)


def _apply_clip(series: pd.Series, lo: float | None, hi: float | None) -> tuple[pd.Series, dict[str, float]]:
    s = pd.to_numeric(series, errors="coerce")
    before = s.copy()
    lower_clipped = 0
    upper_clipped = 0
    if lo is not None:
        lower_clipped = int((s < lo).fillna(False).sum())
        s = s.clip(lower=lo)
    if hi is not None:
        upper_clipped = int((s > hi).fillna(False).sum())
        s = s.clip(upper=hi)
    stats = {
        "n_values_clipped_lower": float(lower_clipped),
        "n_values_clipped_upper": float(upper_clipped),
        "min_before": float(before.min()) if len(before.dropna()) else np.nan,
        "max_before": float(before.max()) if len(before.dropna()) else np.nan,
        "min_after": float(s.min()) if len(s.dropna()) else np.nan,
        "max_after": float(s.max()) if len(s.dropna()) else np.nan,
    }
    return s, stats


def _flip_quantile_columns(df: pd.DataFrame, quantile_cols: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = df.copy()
    missing_required: list[str] = []
    missing_optional: list[str] = []
    for a, b in MANDATORY_SYMMETRIC_PAIRS:
        if a == b:
            if a not in quantile_cols:
                missing_required.append(f"{a}<->{b}")
            continue
        if a not in quantile_cols or b not in quantile_cols:
            missing_required.append(f"{a}<->{b}")
    if missing_required:
        raise ValueError(f"Missing required quantile symmetric pairs for sign-flip target: {missing_required}")
    for a, b in OPTIONAL_SYMMETRIC_PAIRS:
        has_a = a in quantile_cols
        has_b = b in quantile_cols
        if has_a and has_b:
            pass
        elif has_a or has_b:
            missing_optional.append(f"{a}<->{b}")
    original = {c: pd.to_numeric(out[c], errors="coerce") for c in quantile_cols if c in out.columns}
    # Q(-X,q) = -Q(X,1-q)
    mapping = {
        "p01": "p99",
        "p05": "p95",
        "p10": "p90",
        "p30": "p70",
        "p50": "p50",
        "p70": "p30",
        "p90": "p10",
        "p95": "p05",
        "p99": "p01",
    }
    for q, src in mapping.items():
        if q in quantile_cols and src in quantile_cols and src in original:
            out[q] = -original[src]
    return out, missing_required, missing_optional


def canonicalize_quantile_dict(
    qdict: dict[str, pd.Series], target_name: str, *, target_value_mode: str | None = None
) -> dict[str, pd.Series]:
    spec = get_target_postprocessing_spec(target_name, target_value_mode=target_value_mode)
    out = {k: pd.to_numeric(v, errors="coerce") for k, v in qdict.items()}
    if spec.sign_transform == "multiply_by_minus_one":
        out = {k: -v for k, v in out.items()}
    if spec.quantile_flip:
        # expects pXX style keys
        temp = {k: v.copy() for k, v in out.items()}
        pairs = {"p10": "p90", "p30": "p70", "p50": "p50", "p70": "p30", "p90": "p10", "p01": "p99", "p05": "p95", "p95": "p05", "p99": "p01"}
        for q, src in pairs.items():
            if q in out and src in temp:
                out[q] = temp[src]
    for k in list(out.keys()):
        out[k], _ = _apply_clip(out[k], spec.clip_lower, spec.clip_upper)
    return out


def canonicalize_truth_series(series: pd.Series, target_name: str, *, target_value_mode: str | None = None) -> pd.Series:
    spec = get_target_postprocessing_spec(target_name, target_value_mode=target_value_mode)
    s = pd.to_numeric(series, errors="coerce")
    if spec.sign_transform == "multiply_by_minus_one":
        s = -s
    s, _ = _apply_clip(s, spec.clip_lower, spec.clip_upper)
    return s


def canonicalize_prediction_frame(
    df: pd.DataFrame,
    target_name: str,
    quantile_cols: list[str],
    predicted_value_col: str = "predicted_value",
    target_value_mode: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_target_postprocessing_spec(target_name, target_value_mode=target_value_mode)
    out = df.copy()
    report: dict[str, Any] = {
        "target": target_name,
        "transformation_applied": 1.0,
        "sign_flipped": float(spec.sign_transform == "multiply_by_minus_one"),
        "quantiles_flipped": float(spec.quantile_flip),
        "clip_lower": spec.clip_lower if spec.clip_lower is not None else np.nan,
        "clip_upper": spec.clip_upper if spec.clip_upper is not None else np.nan,
        "missing_required_quantile_pairs": "",
        "missing_optional_quantile_pairs": "",
        "status": "ok",
        "target_value_mode": str(target_value_mode or "raw_signed_legacy"),
    }
    crossing_before = np.nan
    crossing_after_prerepair = np.nan
    q_existing = [q for q in quantile_cols if q in out.columns]
    if q_existing:
        arr0 = np.vstack([pd.to_numeric(out[q], errors="coerce").to_numpy(dtype=float) for q in q_existing])
        crossing_before = float(np.mean(np.any(np.diff(arr0, axis=0) < 0, axis=0)))
    if spec.sign_transform == "multiply_by_minus_one":
        if predicted_value_col in out.columns:
            out[predicted_value_col] = -pd.to_numeric(out[predicted_value_col], errors="coerce")
        # Quantile sign handling is done via symmetric quantile flip for sign-flipped targets.
        if not spec.quantile_flip:
            for c in q_existing:
                out[c] = -pd.to_numeric(out[c], errors="coerce")
        elif "p50" not in q_existing and "p50" in out.columns:
            # If p50 exists outside quantile set, flip once here.
            out["p50"] = -pd.to_numeric(out["p50"], errors="coerce")
    if spec.quantile_flip:
        out, missing_req, missing_opt = _flip_quantile_columns(out, quantile_cols=q_existing)
        report["missing_required_quantile_pairs"] = ",".join(missing_req)
        report["missing_optional_quantile_pairs"] = ",".join(missing_opt)
    # clip predicted and quantiles
    clip_stats_acc = {"n_values_clipped_lower": 0.0, "n_values_clipped_upper": 0.0, "min_before": np.nan, "max_before": np.nan, "min_after": np.nan, "max_after": np.nan}
    clip_cols = [c for c in [predicted_value_col] + q_existing if c in out.columns]
    mins_b, maxs_b, mins_a, maxs_a = [], [], [], []
    for c in clip_cols:
        out[c], st = _apply_clip(pd.to_numeric(out[c], errors="coerce"), spec.clip_lower, spec.clip_upper)
        clip_stats_acc["n_values_clipped_lower"] += st["n_values_clipped_lower"]
        clip_stats_acc["n_values_clipped_upper"] += st["n_values_clipped_upper"]
        mins_b.append(st["min_before"]); maxs_b.append(st["max_before"]); mins_a.append(st["min_after"]); maxs_a.append(st["max_after"])
    if mins_b:
        clip_stats_acc["min_before"] = float(np.nanmin(mins_b))
        clip_stats_acc["max_before"] = float(np.nanmax(maxs_b))
        clip_stats_acc["min_after"] = float(np.nanmin(mins_a))
        clip_stats_acc["max_after"] = float(np.nanmax(maxs_a))
    if q_existing:
        arr1 = np.vstack([pd.to_numeric(out[q], errors="coerce").to_numpy(dtype=float) for q in q_existing])
        crossing_after_prerepair = float(np.mean(np.any(np.diff(arr1, axis=0) < 0, axis=0)))
    report.update(clip_stats_acc)
    report["crossing_before_postprocess"] = crossing_before
    report["crossing_after_postprocess_before_repair"] = crossing_after_prerepair
    return out, report


def prediction_target_for_truth_column(truth_col: str) -> str | None:
    return TRUTH_TO_PRED_TARGET.get(str(truth_col))
