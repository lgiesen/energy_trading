from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


_CANONICAL_TARGETS: dict[str, dict[str, object]] = {
    "da": {
        "canonical_target": "da_price",
        "prediction_aliases": ["da_price", "target_da_price", "pred_da_price"],
        "truth_candidates": ["da_price"],
    },
    "afrr_capacity_pos": {
        "canonical_target": "afrr_capacity_price_pos",
        "prediction_aliases": ["target_afrr_capacity_price_pos", "afrr_capacity_price_pos", "pred_afrr_capacity_price_pos"],
        "truth_candidates": ["afrr_capacity_price_pos", "target_afrr_capacity_price_pos"],
    },
    "afrr_capacity_neg": {
        "canonical_target": "afrr_capacity_price_neg",
        "prediction_aliases": ["target_afrr_capacity_price_neg", "afrr_capacity_price_neg", "pred_afrr_capacity_price_neg"],
        "truth_candidates": ["afrr_capacity_price_neg", "target_afrr_capacity_price_neg"],
    },
    "afrr_activation_price_pos": {
        "canonical_target": "afrr_activation_price_vwap_pos",
        "prediction_aliases": [
            "target_afrr_activation_price_vwap_pos",
            "afrr_activation_price_vwap_pos",
            "pred_afrr_activation_price_pos",
        ],
        "truth_candidates": ["afrr_activation_price_vwap_pos", "target_afrr_activation_price_vwap_pos"],
    },
    "afrr_activation_price_neg": {
        "canonical_target": "afrr_activation_price_vwap_neg",
        "prediction_aliases": [
            "target_afrr_activation_price_vwap_neg",
            "afrr_activation_price_vwap_neg",
            "pred_afrr_activation_price_neg",
        ],
        "truth_candidates": ["afrr_activation_price_vwap_neg", "target_afrr_activation_price_vwap_neg"],
    },
    "afrr_activation_rate_pos": {
        "canonical_target": "activation_rate_phys_pos",
        "prediction_aliases": ["target_afrr_activation_rate_pos", "activation_rate_phys_pos", "pred_afrr_activation_rate_pos"],
        "truth_candidates": ["activation_rate_phys_pos", "target_afrr_activation_rate_pos"],
    },
    "afrr_activation_rate_neg": {
        "canonical_target": "activation_rate_phys_neg",
        "prediction_aliases": ["target_afrr_activation_rate_neg", "activation_rate_phys_neg", "pred_afrr_activation_rate_neg"],
        "truth_candidates": ["activation_rate_phys_neg", "target_afrr_activation_rate_neg"],
    },
}


@dataclass(frozen=True)
class TruthMappingResult:
    canonical_target: str
    prediction_target_name: str
    truth_column: str | None
    truth_source_path: str
    status: str
    reason: str


def _find_target_group(prediction_target_name: str) -> str:
    name = str(prediction_target_name).strip()
    for group, meta in _CANONICAL_TARGETS.items():
        aliases: Sequence[str] = meta["prediction_aliases"]  # type: ignore[assignment]
        if name in aliases:
            return group
    known = sorted({a for m in _CANONICAL_TARGETS.values() for a in m["prediction_aliases"]})
    raise ValueError(
        f"Unknown prediction target '{prediction_target_name}'. Known aliases include: {known}"
    )


def resolve_truth_mapping(
    *,
    prediction_target_name: str,
    available_truth_columns: Sequence[str],
    truth_source_path: str | Path,
    fail_on_missing_truth: bool = True,
) -> TruthMappingResult:
    group = _find_target_group(prediction_target_name)
    meta = _CANONICAL_TARGETS[group]
    candidates: list[str] = list(meta["truth_candidates"])  # type: ignore[arg-type]
    found = [c for c in candidates if c in set(available_truth_columns)]

    if len(found) == 1:
        return TruthMappingResult(
            canonical_target=str(meta["canonical_target"]),
            prediction_target_name=str(prediction_target_name),
            truth_column=found[0],
            truth_source_path=str(Path(truth_source_path)),
            status="ok",
            reason="resolved_unique_candidate",
        )

    if len(found) > 1:
        raise ValueError(
            "Ambiguous truth mapping for "
            f"'{prediction_target_name}' (canonical={meta['canonical_target']}): found={found}, "
            f"candidates={candidates}. Keep exactly one canonical truth column in the selected source."
        )

    reason = (
        f"missing_truth_for_target: candidates={candidates}; "
        f"available_columns_sample={list(available_truth_columns)[:30]}"
    )
    if fail_on_missing_truth:
        raise ValueError(
            f"Truth mapping failed for '{prediction_target_name}'. {reason}. "
            "Recommended fix: provide the canonical truth column in --truth-source or update mapping config."
        )
    return TruthMappingResult(
        canonical_target=str(meta["canonical_target"]),
        prediction_target_name=str(prediction_target_name),
        truth_column=None,
        truth_source_path=str(Path(truth_source_path)),
        status="missing",
        reason=reason,
    )
