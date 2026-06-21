"""Shared target and model ordering for RQ1 thesis figures and tables."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


GROUP_ORDER = {
    "da": 0,
    "da_price": 0,
    "da price": 0,
    "afrr_capacity": 1,
    "afrr_capacity_price": 1,
    "afrr capacity": 1,
    "afrr capacity price": 1,
    "afrr_activation_price": 2,
    "afrr activation price": 2,
    "activation price": 2,
    "afrr_activation_rate": 3,
    "afrr activation rate": 3,
    "activation rate": 3,
}

MODEL_LABEL_ORDER = {
    "rlqr": 0,
    "linear": 0,
    "xgb": 1,
    "xgboost": 1,
    "tft": 2,
}

MODEL_LABELS = ["RLQR", "XGB", "TFT"]


def _norm(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    text = re.sub(r"^(pred|target)_", "", text)
    text = text.replace("vwap", "")
    text = re.sub(r"[_\\/-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def target_group_sort_key(value: Any) -> tuple[int, str]:
    text = _norm(value)
    compact = text.replace(" ", "_")
    for key, rank in GROUP_ORDER.items():
        if key in {text, compact} or key in text or key in compact:
            return rank, text
    return 99, text


def target_sign_sort_key(value: Any) -> int:
    raw = str(value or "").strip()
    raw_no_space = re.sub(r"\s+", "", raw)
    if raw_no_space.endswith("+"):
        return 0
    if raw_no_space.endswith("-") or raw_no_space.endswith("\u2212"):
        return 1
    text = _norm(value)
    parts = set(text.split())
    if "positive" in parts or "pos" in parts or text.endswith(" pos"):
        return 0
    if "negative" in parts or "neg" in parts or text.endswith(" neg"):
        return 1
    return 0


def target_sort_key(value: Any) -> tuple[int, int, str]:
    text = _norm(value)
    group_rank, _ = target_group_sort_key(text)
    return group_rank, target_sign_sort_key(value), text


def sort_target_frame(df: pd.DataFrame, *, target_col: str = "target", extra_cols: list[str] | None = None) -> pd.DataFrame:
    if df.empty or target_col not in df.columns:
        return df.copy()
    extra_cols = extra_cols or []
    out = df.copy()
    keys = out[target_col].map(target_sort_key)
    out["_rq1_target_group_order"] = keys.map(lambda x: x[0])
    out["_rq1_target_sign_order"] = keys.map(lambda x: x[1])
    out["_rq1_target_label_order"] = keys.map(lambda x: x[2])
    internal_cols = ["_rq1_target_group_order", "_rq1_target_sign_order", "_rq1_target_label_order"]
    sort_cols = [*internal_cols, *extra_cols]
    return out.sort_values(sort_cols).drop(columns=internal_cols).reset_index(drop=True)


def model_sort_key(value: Any) -> tuple[int, str]:
    text = _norm(value)
    compact = text.replace(" ", "_")
    return MODEL_LABEL_ORDER.get(compact, MODEL_LABEL_ORDER.get(text, 99)), text


def ordered_model_labels(values: Any | None = None) -> list[str]:
    if values is None:
        return MODEL_LABELS.copy()
    seen: dict[str, str] = {}
    for value in values:
        label = str(value)
        if label not in seen:
            seen[label] = label
    return sorted(seen.values(), key=model_sort_key)


def sort_model_frame(df: pd.DataFrame, *, model_col: str = "model_label", extra_cols: list[str] | None = None) -> pd.DataFrame:
    if df.empty or model_col not in df.columns:
        return df.copy()
    extra_cols = extra_cols or []
    out = df.copy()
    keys = out[model_col].map(model_sort_key)
    out["_rq1_model_order"] = keys.map(lambda x: x[0])
    out["_rq1_model_label_order"] = keys.map(lambda x: x[1])
    sort_cols = ["_rq1_model_order", "_rq1_model_label_order", *extra_cols]
    return out.sort_values(sort_cols).drop(columns=["_rq1_model_order", "_rq1_model_label_order"]).reset_index(drop=True)


def ordered_unique(values: Any, *, group: bool = False) -> list[Any]:
    seen: dict[str, Any] = {}
    for value in values:
        key = str(value)
        if key not in seen:
            seen[key] = value
    sorter = target_group_sort_key if group else target_sort_key
    return sorted(seen.values(), key=sorter)
