"""Target-specific feature routing and model regularization policies."""

from __future__ import annotations

import re
from typing import Any

BundleName = str

# Feature variant policy discovered from ablation runs.
TARGET_FEATURE_VARIANT_MAP: dict[str, str] = {
    "target_da_price": "plus_neighbors",
    "target_afrr_capacity_price_pos": "plus_weather",
    "target_afrr_capacity_price_neg": "plus_neighbors",
    "target_afrr_activation_price_vwap_pos": "core_only",
    "target_afrr_activation_price_vwap_neg": "core_only",
    "target_afrr_activation_rate_pos": "core_only",
    "target_afrr_activation_rate_neg": "core_only",
}

# XGBoost target-specific defaults. These are applied per target and override
# global defaults where specified.
TARGET_XGB_PARAMS: dict[str, dict[str, float]] = {
    "target_da_price": {
        "max_depth": 8.0,
        "min_child_weight": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "early_stopping_rounds": 50.0,
    },
    "target_afrr_capacity_price_pos": {
        "max_depth": 5.0,
        "min_child_weight": 6.0,
        "reg_alpha": 1.5,
        "reg_lambda": 5.0,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 30.0,
    },
    "target_afrr_capacity_price_neg": {
        "max_depth": 5.0,
        "min_child_weight": 6.0,
        "reg_alpha": 1.5,
        "reg_lambda": 5.0,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 30.0,
    },
    "target_afrr_activation_price_vwap_pos": {
        "max_depth": 4.0,
        "min_child_weight": 10.0,
        "reg_alpha": 2.0,
        "reg_lambda": 8.0,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "early_stopping_rounds": 20.0,
    },
    "target_afrr_activation_price_vwap_neg": {
        "max_depth": 4.0,
        "min_child_weight": 10.0,
        "reg_alpha": 2.0,
        "reg_lambda": 8.0,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "early_stopping_rounds": 20.0,
    },
    "target_afrr_activation_rate_pos": {
        "max_depth": 3.0,
        "min_child_weight": 12.0,
        "reg_alpha": 2.5,
        "reg_lambda": 10.0,
        "subsample": 0.75,
        "colsample_bytree": 0.7,
        "early_stopping_rounds": 20.0,
    },
    "target_afrr_activation_rate_neg": {
        "max_depth": 3.0,
        "min_child_weight": 12.0,
        "reg_alpha": 2.5,
        "reg_lambda": 10.0,
        "subsample": 0.75,
        "colsample_bytree": 0.7,
        "early_stopping_rounds": 20.0,
    },
}

# TFT target-specific defaults. For aFRR targets, dropout and early-stopping
# are tightened to reduce overfitting on noisy labels.
TARGET_TFT_PARAMS: dict[str, dict[str, float]] = {
    "target_da_price": {
        "dropout": 0.1,
        "early_stopping_patience": 15.0,
        "max_epochs": 120.0,
    },
    "target_afrr_capacity_price_pos": {
        "dropout": 0.25,
        "early_stopping_patience": 8.0,
        "max_epochs": 220.0,
    },
    "target_afrr_capacity_price_neg": {
        "dropout": 0.25,
        "early_stopping_patience": 8.0,
        "max_epochs": 220.0,
    },
    "target_afrr_activation_price_vwap_pos": {
        "dropout": 0.3,
        "early_stopping_patience": 6.0,
        "max_epochs": 220.0,
    },
    "target_afrr_activation_price_vwap_neg": {
        "dropout": 0.3,
        "early_stopping_patience": 6.0,
        "max_epochs": 220.0,
    },
    "target_afrr_activation_rate_pos": {
        "dropout": 0.35,
        "early_stopping_patience": 6.0,
        "max_epochs": 220.0,
    },
    "target_afrr_activation_rate_neg": {
        "dropout": 0.35,
        "early_stopping_patience": 6.0,
        "max_epochs": 220.0,
    },
}


def _feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    cols = list(feature_cols)
    core_patterns = [
        r"_lag_(1|2|3|6|12|24|48|168)h$",
        r"^(hour|dayofweek|weekday|month)_(sin|cos)$",
        r"^is_(weekend|morning|afternoon|evening|night|bridge_day|christmas_break|payday_period)$",
        r"^da_price_(pit|lag_|diff|mean_|std_|ewma|slog1p)",
        r"^(da_price_pit|market_regime_picasso|is_picasso_active)$",
    ]
    weather_patterns = [
        r"(wind|solar|load_forecast|residual_load_forecast|renewable_share_forecast)",
        r"(temperature|temp|weather)",
    ]
    outages_patterns = [r"(planned_outages|unplanned_outages|outage)"]
    neighbors_patterns = [r"(neighbor_spread|da_spread_de_|da_price_(AT|FR|NL)|cross_border|interconnector)"]

    def pick(patterns: list[str]) -> list[str]:
        rx = re.compile("|".join(patterns))
        return sorted([c for c in cols if rx.search(c)])

    g1 = sorted(set(pick(core_patterns)))
    g2 = sorted(set(g1 + pick(weather_patterns)))
    g3 = sorted(set(g2 + pick(outages_patterns)))
    g4 = sorted(set(g3 + pick(neighbors_patterns)))

    if not g1:
        g1 = sorted([c for c in cols if ("_lag_" in c) or c.endswith("_sin") or c.endswith("_cos")])
    if not g2:
        g2 = g1
    if not g3:
        g3 = g2
    if not g4:
        g4 = g3

    return {
        "core_only": g1,
        "plus_weather": g2,
        "plus_outages": g3,
        "plus_neighbors": g4,
    }


def resolve_feature_columns_for_target(
    feature_columns: list[str],
    target_col: str,
) -> tuple[str, list[str]]:
    groups = _feature_groups(feature_columns)
    variant = TARGET_FEATURE_VARIANT_MAP.get(target_col, "plus_neighbors")
    cols = groups.get(variant, [])
    if not cols:
        cols = feature_columns
        variant = "all_features_fallback"
    return variant, cols


def resolve_xgb_params_for_target(target_col: str, base_params: dict[str, Any]) -> dict[str, Any]:
    policy = TARGET_XGB_PARAMS.get(target_col, {})
    out = dict(base_params)
    out.update(policy)
    return out


def resolve_tft_params_for_target(target_col: str, base_params: dict[str, Any]) -> dict[str, Any]:
    policy = TARGET_TFT_PARAMS.get(target_col, {})
    out = dict(base_params)
    out.update(policy)
    return out
