#!/usr/bin/env python3
"""Generate RQ3 market-strategy comparison outputs.

Figure 1: annualized net profit by market participation strategy.
Figure 2: revenue and cost decomposition by market participation strategy.
Figure 3: cumulative net profit over time by market participation strategy.
Figure 4: operational intensity by market participation strategy.

This script reads existing simulation outputs only. It does not run simulations.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_trading.evaluation.style import MARKET_COLOR_MAP, THESIS_PALETTE, apply_geo_style, thesis_titlecase


DEFAULT_OUT_ROOT = Path("artifacts/benchmark/rq3_market_comparison")
DEFAULT_EXPORT_DIR = (
    "/Users/leori/Desktop/ uni/3 Master IS/25 MA/"
    "MA___Elevated_Energy_Trading__Forecasting_Expected_Profit_From_Participating_in_the_Day_Ahead_and_Balancing_Markets/"
    "figures/4-results/rq3_market_comparison"
)
DEFAULT_INPUT_ROOTS = [
    Path("artifacts/benchmark/rq3_market_comparison"),
    Path("artifacts/benchmark/rq3_simulation_benchmark"),
    Path("artifacts/simulation_runs"),
]
STRATEGY_ORDER = ["Multi", "DA-only", "BCM-only", "BEM-only"]
STRATEGY_COLOR = {
    "Multi": THESIS_PALETTE["primary"],
    "DA-only": THESIS_PALETTE["neutral_dark"],
    "BCM-only": MARKET_COLOR_MAP["BCM capacity"],
    "BEM-only": MARKET_COLOR_MAP["BEM"],
}
RQ3_MARKET_ORDER = ["DA", "BCM", "BEM"]
RQ3_MARKET_COLOR = {
    "DA": MARKET_COLOR_MAP.get("DA", THESIS_PALETTE["primary"]),
    "BCM": MARKET_COLOR_MAP.get("BCM capacity", THESIS_PALETTE["secondary"]),
    "BEM": MARKET_COLOR_MAP.get("BEM", THESIS_PALETTE["tertiary"]),
}
RQ3_SINGLE_MARKET_STRATEGY = {
    "DA": "DA-only",
    "BCM": "BCM-only",
    "BEM": "BEM-only",
}
STRATEGY_GAMUT = {
    "multi": "Multi",
    "multi-market": "Multi",
    "multimarket": "Multi",
    "multi_market": "Multi",
    "da": "DA-only",
    "da_only": "DA-only",
    "da-only": "DA-only",
    "day-ahead": "DA-only",
    "dayahead": "DA-only",
    "bcm": "BCM-only",
    "bcm_only": "BCM-only",
    "bcm-only": "BCM-only",
    "afrr": "BCM-only",
    "afrr-only": "BCM-only",
    "bem": "BEM-only",
    "bem_only": "BEM-only",
    "bem-only": "BEM-only",
}
PROFIT_COLS = [
    "annualized_realized_net_revenue_eur",
    "annualized_realized_net_profit_eur",
    "annualized_net_profit_eur_per_year",
    "annualized_profit_eur_per_year",
    "annualized_realized_pnl_eur",
]
NET_PROFIT_COLS = [
    "realized_net_revenue_eur",
    "realized_net_profit_eur",
    "realized_total_pnl_eur",
    "pnl_real_eur",
]
REVENUE_COMPONENT_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "DA revenue",
        (
            "annualized_da_revenue_eur",
            "annualized_da_net_revenue_eur",
            "annualized_da_pnl_eur",
            "da_revenue_eur",
            "da_net_revenue_eur",
            "da_pnl_eur",
            "real_revenue_da_eur",
            "realized_da_revenue_eur",
            "realized_da_net_revenue_eur",
        ),
    ),
    (
        "ID revenue",
        (
            "annualized_id_revenue_eur",
            "annualized_id_net_revenue_eur",
            "id_net_revenue_eur",
            "id_revenue_eur",
        ),
    ),
    (
        "BCM capacity revenue",
        (
            "annualized_bcm_capacity_revenue_eur",
            "bcm_capacity_revenue_eur",
            "afrr_capacity_revenue_eur",
        ),
    ),
    (
        "BCM-linked BEM activation revenue",
        (
            "annualized_bcm_activation_revenue_eur",
            "bcm_linked_activation_revenue_eur",
            "bcm_activation_revenue_eur",
        ),
    ),
    (
        "BEM activation revenue",
        (
            "annualized_bem_activation_revenue_eur",
            "annualized_afrr_activation_revenue_eur",
            "bem_activation_revenue_eur",
            "afrr_activation_revenue_eur",
        ),
    ),
)
COST_DETAIL_COMPONENT_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Degradation cost",
        (
            "annualized_realized_degradation_cost_eur",
            "annualized_degradation_cost_eur",
            "realized_degradation_cost_eur",
        ),
    ),
    (
        "Auxiliary cost",
        (
            "annualized_realized_aux_cost_eur",
            "annualized_aux_cost_eur",
            "realized_aux_cost_eur",
            "aux_cost_eur",
        ),
    ),
    ("Transaction cost", ("annualized_transaction_cost_eur", "transaction_cost_eur")),
    (
        "Activation penalty",
        (
            "annualized_penalty_cost_eur",
            "penalty_cost_eur",
            "annualized_bem_penalty_cost_eur",
            "annualized_bem_activation_cost_eur",
            "bem_penalty_cost_eur",
            "bem_activation_cost_eur",
        ),
    ),
    ("Other penalties", ("annualized_other_penalty_cost_eur", "other_penalty_cost_eur")),
    ("ID recourse cost", ("annualized_id_recourse_cost_eur", "id_recourse_cost_eur")),
    ("Terminal SoC repair", ("annualized_terminal_soc_repair_cost_eur", "terminal_soc_repair_cost_eur")),
)
COST_TOTAL_CANDIDATES = ("annualized_total_costs_eur", "total_costs_eur")
STRUCTURALLY_ZERO_COMPONENTS = {
    "DA-only": ("BCM capacity revenue", "BEM activation revenue", "BCM-linked BEM activation revenue"),
    "BCM-only": ("DA revenue", "BEM activation revenue", "ID revenue"),
    "BEM-only": ("DA revenue", "BCM capacity revenue", "ID revenue"),
}
REVENUE_LABEL = "Revenue"
COST_LABEL = "Costs"
REVENUE_COMPONENT_COLOR = {
    "DA revenue": MARKET_COLOR_MAP.get("DA", THESIS_PALETTE["primary"]),
    "BCM capacity revenue": MARKET_COLOR_MAP.get("BCM capacity", THESIS_PALETTE["secondary"]),
    "BCM-linked BEM activation revenue": MARKET_COLOR_MAP.get("BCM activation", THESIS_PALETTE["secondary"]),
    "BEM activation revenue": MARKET_COLOR_MAP.get("BEM", THESIS_PALETTE["tertiary"]),
    "ID revenue": MARKET_COLOR_MAP.get("ID", THESIS_PALETTE["neutral_dark"]),
}
COST_COMPONENT_COLOR = {
    "Degradation cost": "#F0746E",
    "Auxiliary cost": "#DC3977",
    "Transaction cost": "#7A7A7A",
    "Activation penalty": "#333333",
    "Offer cost": "#2E7D32",
    "ID recourse cost": "#5B4B8A",
    "Activation cost": "#0D4A70",
    "Terminal SoC repair": "#045275",
    "Total costs": "#666666",
}
REVENUE_COMPONENTS = tuple(name for name, _ in REVENUE_COMPONENT_SPECS)
COST_COMPONENTS = tuple(name for name, _ in COST_DETAIL_COMPONENT_SPECS) + ("Total costs",)
OPERATIONAL_METRIC_SPECS: tuple[dict[str, Any], ...] = (
    {
        "metric": "Throughput",
        "display_metric": "Throughput in GWh",
        "unit": "GWh/year",
        "candidates": ("throughput_mwh_total", "total_throughput_mwh"),
        "annualize": True,
        "scale": 1.0 / 1000.0,
    },
    {
        "metric": "Equivalent full cycles",
        "unit": "cycles",
        "candidates": ("equivalent_full_cycles_total", "total_equivalent_full_cycles", "equivalent_full_cycles"),
    },
    {
        "metric": "Avg daily cycles",
        "unit": "cycles/day",
        "candidates": ("equivalent_full_cycles_per_day", "average_daily_cycles"),
    },
    {
        "metric": "Mean SoC",
        "display_metric": "Mean SoC in MWh",
        "unit": "MWh",
        "candidates": ("mean_soc_mwh", "realized_mean_soc_mwh", "soc_mwh_mean"),
    },
    {
        "metric": "Auxiliary cost",
        "display_metric": "Auxiliary cost in kEUR",
        "unit": "kEUR/year",
        "candidates": ("realized_aux_cost_eur", "aux_cost_eur"),
        "annualize": True,
        "scale": 1.0 / 1000.0,
    },
    {
        "metric": "Degradation cost",
        "display_metric": "Degra-\ndation cost\nin kEUR",
        "unit": "kEUR/year",
        "candidates": (
            "realized_degradation_cost_eur",
            "degradation_cost_eur",
        ),
        "annualize": True,
        "scale": 1.0 / 1000.0,
    },
    {
        "metric": "ID recourse volume",
        "display_metric": "ID recourse volume in MWh",
        "unit": "MWh/year",
        "candidates": ("id_abs_mwh_total", "id_recourse_mwh_total", "id_buy_mwh_total+id_sell_mwh_total"),
        "annualize": True,
    },
    {
        "metric": "Fallback count",
        "unit": "count",
        "candidates": ("fallback_optimization_count", "optimizer_fallback_count", "fallback_used"),
    },
    {
        "metric": "Fallback share",
        "unit": "%",
        "candidates": ("fallback_optimization_share", "fallback_share"),
    },
    {
        "metric": "Infeasibility count",
        "unit": "count",
        "candidates": ("combined_infeasibility_hours", "infeasibility_count", "optimization_infeasible_count"),
    },
    {
        "metric": "Infeasibility share",
        "unit": "%",
        "candidates": ("combined_infeasibility_hours_share", "infeasibility_share"),
    },
    {
        "metric": "Missed activations",
        "unit": "count",
        "candidates": ("missed_activation_count", "missed_activation_events"),
    },
    {
        "metric": "SoC violation count",
        "unit": "count",
        "candidates": ("soc_violation_hours", "protected_soc_violation_count", "physical_soc_violation_count"),
    },
    {
        "metric": "Reserve-headroom violations",
        "unit": "count",
        "candidates": ("reserve_headroom_shortfall_hours", "reserve_headroom_shortfall_count"),
    },
)


@dataclass(frozen=True)
class Candidate:
    root: Path
    source_files: tuple[Path, ...]
    rows: pd.DataFrame
    score: tuple[int, int, float]


_QUANTILE_PATTERN = re.compile(r"(p\d+)", re.IGNORECASE)
TS_TIMESTAMP_CANDIDATES: tuple[str, ...] = (
    "timestamp_utc",
    "timestamp",
    "time_utc",
    "time",
    "delivery_utc",
    "delivery_start_utc",
    "delivery_start",
    "date",
    "datetime",
)
TS_STRATEGY_CANDIDATES: tuple[str, ...] = ("trading_strategy", "strategy", "market_strategy", "strategy_name")
TS_MODEL_CANDIDATES: tuple[str, ...] = ("model", "model_key", "model_name")
TS_QUANTILE_CANDIDATES: tuple[str, ...] = (
    "quantile_pair",
    "quantile",
    "quantile_low",
    "quantile_high",
    "scenario",
)
TS_CUMULATIVE_PNL_CANDS: tuple[str, ...] = (
    "cumulative_net_profit_eur",
    "cumulative_realized_net_profit_eur",
    "cumulative_pnl_eur",
    "cumulative_realized_pnl_eur",
    "cumulative_net_revenue_eur",
    "cumulative_profit_eur",
)
TS_PERIOD_PNL_CANDS: tuple[str, ...] = (
    "realized_net_profit_eur",
    "realized_pnl_eur",
    "net_profit_eur",
    "pnl_eur",
    "net_revenue_eur",
    "profit_eur",
)


def _safe_float(value: Any) -> float:
    x = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(x) if pd.notna(x) and math.isfinite(float(x)) else math.nan


def _normalize_for_matching(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def _resolve_matching_column(columns: list[str], candidates: tuple[str, ...], prefer_prefix: str | None = None) -> str | None:
    normalized_columns = {_normalize_for_matching(c): c for c in columns}
    norm_candidates = [_normalize_for_matching(c) for c in candidates]
    preferred: str | None = None
    for c in columns:
        norm = _normalize_for_matching(c)
        if prefer_prefix is not None and norm.startswith(_normalize_for_matching(prefer_prefix)):
            preferred = c
            break
        if norm in norm_candidates:
            return c
    # Fuzzy fallback: token match.
    for cand_norm, raw in zip(norm_candidates, candidates):
        for col_norm, col_raw in normalized_columns.items():
            if cand_norm and (cand_norm in col_norm or col_norm in cand_norm):
                if preferred is None:
                    preferred = col_raw
                if prefer_prefix is None:
                    return col_raw
    return preferred


def _resolve_exact_matching_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized_columns = {_normalize_for_matching(c): c for c in columns}
    for candidate in candidates:
        col = normalized_columns.get(_normalize_for_matching(candidate))
        if col is not None:
            return col
    return None


def _latex_escape(value: Any) -> str:
    s = str(value)
    replacements = {
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
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def _tex_float(value: float, digits: int = 3) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def _tex_color_def(name: str, hex_color: str) -> str:
    hex_color = str(hex_color).lstrip("#")
    red = int(hex_color[0:2], 16) / 255.0
    green = int(hex_color[2:4], 16) / 255.0
    blue = int(hex_color[4:6], 16) / 255.0
    return rf"\definecolor{{{name}}}{{rgb}}{{{red:.4f},{green:.4f},{blue:.4f}}}"


def _revenue_segment_label(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):,.0f}"


def _revenue_label_position(strategy: str, component: str) -> tuple[float, float, str, str]:
    """Return x/y offsets and horizontal alignment for revenue value labels."""
    if strategy == "Multi" and component == "DA revenue":
        return 0.44, 0.0, "left", "west"
    if strategy == "BCM-only" and component == "BCM-linked BEM activation revenue":
        return 0.44, 64.0, "left", "west"
    if component == "BCM-linked BEM activation revenue":
        return 0.44, 12.0, "left", "west"
    if strategy == "BCM-only" and component == "BCM capacity revenue":
        return 0.44, 10.0, "left", "west"
    return 0.0, 0.0, "center", "center"


def _revenue_label_text_color(component: str, anchor: str) -> str:
    if anchor == "west":
        return "black"
    if component in {"DA revenue", "BEM activation revenue"}:
        return "white"
    return "black"


def _revenue_label_tex_xshift(anchor: str) -> str:
    """Keep side labels close to the bar edge in PGFPlots point units."""
    return "11pt" if anchor == "west" else "0pt"


def _revenue_cost_strategy_axis_label(strategy: str) -> str:
    return "Multi-market" if strategy == "Multi" else strategy


def normalize_strategy(value: Any, source_path: Path | None = None) -> str | None:
    candidates = [str(value or "")]
    if source_path is not None:
        candidates.extend(source_path.parts)
    joined = " ".join(candidates).lower().replace("_", "-")
    tokens = {part.lower().replace("_", "-") for part in candidates if str(part).strip()}
    if any(token in {"multi", "multi-market", "multimarket"} for token in tokens) or "multi" in joined:
        return "Multi"
    if any(token in {"da", "da-only", "day-ahead"} for token in tokens) or "da-only" in joined:
        return "DA-only"
    if any(token in {"bcm", "bcm-only", "afrr", "afrr-only"} for token in tokens) or "bcm-only" in joined:
        return "BCM-only"
    if any(token in {"bem", "bem-only"} for token in tokens) or "bem-only" in joined:
        return "BEM-only"
    return None


def _format_value(value_eur: float) -> str:
    return "n/a" if not math.isfinite(value_eur) else f"{value_eur / 1000.0:,.0f}"


def _candidate_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    names = {
        "performance_metrics.csv",
        "performance_metrics_all_scenarios.csv",
        "strategy_overview.csv",
        "strategy_overview_valid_only.csv",
        "rq3_market_strategy_summary.csv",
        "annualized_net_profit_by_market_strategy.csv",
        "performance_metric_reconciliation_debug_all.csv",
        "backtest_summary.json",
        "quantile_sweep_summary.csv",
        "quantile_sweep_summary.json",
    }
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name in names and p.suffix.lower() in {".csv", ".json"})


def _collect_ts_candidates(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in {".csv", ".parquet"} else []
    if not root.exists():
        return []
    names = {
        "cumulative_net_profit_by_market_strategy.csv",
        "net_profit_by_market_strategy.csv",
        "cumulative_net_profit.csv",
        "net_profit.csv",
        "market_strategy_timeseries.csv",
        "daily_performance_metrics.csv",
        "daily_performance_metrics_all_scenarios.csv",
        "market_strategy_backtest_hourly.csv",
        "market_strategy_backtest_hourly.parquet",
        "backtest_hourly.csv",
        "backtest_hourly.parquet",
        "hourly_pnl.csv",
        "hourly_pnl.parquet",
        "hourly_results.csv",
        "hourly_results.parquet",
    }
    candidates = [p for p in root.rglob("*.csv") if p.name in names]
    candidates.extend(p for p in root.rglob("*.parquet") if p.name in names)
    if not candidates:
        # Fallback: all known parquet/csv files in result sections and per-strategy folders.
        candidates = sorted(
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".csv", ".parquet"}
            and any(
                key in p.name.lower()
                for key in (
                    "hourly",
                    "daily",
                    "profit",
                    "cumulative",
                    "net",
                    "pnl",
                    "strategy",
                    "strategy_summary",
                )
            )
        )
    return sorted(candidates)


def _ts_source_priority(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    # Prefer daily period PnL; it matches the thesis cumulative comparison best
    # and avoids excessive hourly tick density.
    if name == "daily_performance_metrics.csv":
        return (0, len(path.parts), str(path))
    if name == "daily_performance_metrics_all_scenarios.csv":
        return (1, len(path.parts), str(path))
    if name == "performance_paths_long.csv":
        return (2, len(path.parts), str(path))
    if "hourly" in name or name.endswith(".parquet"):
        return (3, len(path.parts), str(path))
    return (4, len(path.parts), str(path))


def _duration_days(row: pd.Series) -> tuple[float, str]:
    for col in ["analysis_days", "n_days", "test_period_days", "duration_days"]:
        if col in row.index:
            value = _safe_float(row[col])
            if value > 0:
                return value, col
    for start_col, end_col in [("start_utc", "end_utc"), ("start", "end")]:
        if start_col in row.index and end_col in row.index:
            start = pd.to_datetime(row[start_col], errors="coerce", utc=True)
            end = pd.to_datetime(row[end_col], errors="coerce", utc=True)
            if pd.notna(start) and pd.notna(end) and end > start:
                return float((end - start).total_seconds() / 86400.0), f"{start_col}/{end_col}"
    return math.nan, ""


def _normalize_strategy_from_file(path: Path) -> str | None:
    for part in path.parents:
        candidate = normalize_strategy(part.name)
        if candidate is not None:
            return candidate
    return None


def _extract_model_quantile_hint(path: Path, row: pd.Series | None = None) -> tuple[str | None, str | None]:
    candidate_model = None
    candidate_quantile = None
    text = " ".join(part.lower() for part in path.parts)
    if "xgb" in text or "xgboost" in text:
        candidate_model = "xgb"
    elif "tft" in text:
        candidate_model = "tft"
    elif "linear" in text:
        candidate_model = "linear"
    elif "rlqr" in text:
        candidate_model = "rlqr"
    elif row is not None:
        raw_model = row.get("model", row.get("model_key", row.get("model_name")))
        if pd.notna(raw_model):
            raw_model = str(raw_model).lower()
            if "xgb" in raw_model or "xgboost" in raw_model:
                candidate_model = "xgb"
            elif "tft" in raw_model:
                candidate_model = "tft"
            elif "linear" in raw_model:
                candidate_model = "linear"
            elif "rlqr" in raw_model:
                candidate_model = "rlqr"
    for part in [path.name] + list(path.parts):
        for token in re.findall(_QUANTILE_PATTERN, part.lower()):
            if token:
                candidate_quantile = token.lower()
                break
        if candidate_quantile:
            break
    if row is not None:
        raw_quantile = (
            row.get("quantile_pair")
            or row.get("quantile")
            or row.get("quantile_low")
            or row.get("quantile_high")
            or row.get("scenario")
        )
        if pd.notna(raw_quantile):
            match = _QUANTILE_PATTERN.search(str(raw_quantile))
            if match:
                candidate_quantile = match.group(1).lower()
    return candidate_model, candidate_quantile


def _to_strat_series(
    raw_series: Any,
    strategy_hint: str | None,
) -> str | None:
    if pd.isna(raw_series):
        return strategy_hint
    candidate = str(raw_series).strip()
    normalized = normalize_strategy(candidate)
    if normalized is None:
        normalized = STRATEGY_GAMUT.get(candidate.lower().replace("_", "-"), None)
    return normalized or strategy_hint


def _canonicalize_strategy(value: Any) -> str | None:
    if pd.isna(value):
        return None
    candidate = str(value).strip()
    normalized = _normalize_for_matching(candidate)
    if candidate in STRATEGY_ORDER:
        return candidate
    canonical = STRATEGY_GAMUT.get(candidate.lower(), None)
    if canonical is not None:
        return canonical
    for key, target in STRATEGY_GAMUT.items():
        if normalized == _normalize_for_matching(key):
            return target
    return None


def _read_ts_candidate(path: Path, require_model: str | None = "xgb", require_quantile: str | None = "p50") -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            raise ValueError(f"Could not read candidate RQ3 TS file {path}: {exc}") from exc
    else:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"Could not read candidate RQ3 TS file {path}: {exc}") from exc
    if df.empty:
        return pd.DataFrame()
    columns = list(df.columns)
    time_col = _resolve_matching_column(columns, TS_TIMESTAMP_CANDIDATES)
    if time_col is None:
        return pd.DataFrame()
    model_hint, quantile_hint = _extract_model_quantile_hint(path)
    model_filter = model_hint
    quantile_filter = quantile_hint
    cumulative_col = _resolve_matching_column(columns, TS_CUMULATIVE_PNL_CANDS)
    if cumulative_col is not None:
        cumulative_norm = _normalize_for_matching(cumulative_col)
        if "cumulative" not in cumulative_norm:
            cumulative_col = None
    period_col = _resolve_matching_column(columns, TS_PERIOD_PNL_CANDS)
    strategy_hint = _normalize_strategy_from_file(path)
    strategy_col = _resolve_matching_column(columns, TS_STRATEGY_CANDIDATES)
    quantile_col = _resolve_matching_column(columns, TS_QUANTILE_CANDIDATES)
    model_col = _resolve_matching_column(columns, TS_MODEL_CANDIDATES)
    if cumulative_col is None and period_col is None:
        # Candidate has no direct pnl column; interpret wide columns as strategy rows only if they are numeric and strategy-like.
        wide_value_cols = [
            col
            for col in columns
            if col != time_col and col != strategy_col and col != model_col and col != quantile_col and pd.api.types.is_numeric_dtype(df[col].dtype)
            and any(token in _normalize_for_matching(col) for token in ("multi", "da", "bcm", "afrr", "bem"))
        ]
        if not wide_value_cols:
            return pd.DataFrame()
        value_cols = wide_value_cols
        value_source = "wide"
    else:
        value_cols = [cumulative_col or period_col]
        value_source = "narrow"
    rows: list[dict[str, Any]] = []
    if strategy_col is not None:
        for _, row in df.iterrows():
            model, quantile = _extract_model_quantile_hint(path, row)
            if require_model and model and require_model not in model:
                continue
            if require_quantile and quantile and quantile != require_quantile:
                continue
            strategy = _to_strat_series(row.get(strategy_col), strategy_hint)
            if strategy is None:
                continue
            timestamp = pd.to_datetime(row[time_col], errors="coerce", utc=True)
            if pd.isna(timestamp):
                continue
            for pnl_col in value_cols:
                pnl_value = _safe_float(row.get(pnl_col))
                if not math.isfinite(pnl_value):
                    continue
                rows.append(
                    {
                        "timestamp_utc": timestamp,
                        "strategy": strategy,
                        "raw_pnl_eur": pnl_value,
                        "is_cumulative_input": cumulative_col is not None,
                        "source_file": str(path),
                        "model": model or model_filter or "",
                        "quantile": quantile or quantile_filter or "",
                    }
                )
        return pd.DataFrame(rows)

    # Wide-format fallback: one column per strategy in each row.
    for _, row in df.iterrows():
        model, quantile = _extract_model_quantile_hint(path, row)
        if require_model and model and require_model not in model:
            continue
        if require_quantile and quantile and quantile != require_quantile:
            continue
        timestamp = pd.to_datetime(row[time_col], errors="coerce", utc=True)
        if pd.isna(timestamp):
            continue
        for col in value_cols:
            strategy = _to_strat_series(col, strategy_hint)
            if strategy is None:
                continue
            pnl_value = _safe_float(row[col])
            if not math.isfinite(pnl_value):
                continue
            rows.append(
                {
                    "timestamp_utc": timestamp,
                    "strategy": strategy,
                    "raw_pnl_eur": pnl_value,
                    "is_cumulative_input": value_source == "narrow" and cumulative_col is not None,
                    "source_file": str(path),
                    "model": model or model_filter or "",
                    "quantile": quantile or quantile_filter or "",
                }
            )
    return pd.DataFrame(rows)
def _extract_quantile_tag(*raw_values: Any) -> str:
    for value in raw_values:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        candidate = str(value)
        match = _QUANTILE_PATTERN.search(candidate)
        if match:
            return match.group(1).lower()
        cleaned = candidate.strip().replace("_", "-").replace(" ", "-")
        if cleaned:
            return cleaned.lower()
    return ""


def _annualized_profit(row: pd.Series) -> tuple[float, str, str]:
    for col in PROFIT_COLS:
        if col in row.index:
            value = _safe_float(row[col])
            if math.isfinite(value):
                return value, col, "direct"
    for col in NET_PROFIT_COLS:
        if col not in row.index:
            continue
        net_profit = _safe_float(row[col])
        days, source = _duration_days(row)
        if math.isfinite(net_profit) and days > 0:
            return net_profit * 365.0 / days, f"{col};{source}", "annualized_from_duration"
    return math.nan, "", ""


def _resolve_component(row: pd.Series, candidates: tuple[str, ...]) -> tuple[float, str, str, bool]:
    """Return first available component value from candidates.

    Returns (value, col_used, method, annualized).
    method is one of {"annualized", "annualized_from_duration", "raw", "missing"}.
    annualized indicates whether value already is annualized.
    """
    available: list[str] = []
    for col in candidates:
        if col in row.index:
            available.append(col)
    # Prefer explicitly annualized columns when present.
    for col in available:
        if col.startswith("annualized_"):
            value = _safe_float(row[col])
            if math.isfinite(value):
                return value, col, "annualized", True
    # Raw totals / annualize per duration if needed.
    for col in available:
        if col.startswith("annualized_"):
            continue
        value = _safe_float(row[col])
        if not math.isfinite(value):
            continue
        if col in {"n_days", "test_period_days", "duration_days", "duration_days_in_year", "hours", "n_hours"}:
            continue
        days, source = _duration_days(row)
        if days > 0:
            annualized = value * 365.0 / days
            return annualized, col, f"annualized_from_duration:{source}", False
        return math.nan, col, "missing_duration", False
    # If both explicit annualized and raw candidates are missing, this component is unavailable.
    # Keep a single warning path for clarity.
    return math.nan, ",".join(candidates[:3]) if candidates else "", "missing", False


def _to_plot_value(value: float, scale: str = "kEUR") -> float:
    if not math.isfinite(value):
        return math.nan
    if scale == "kEUR":
        return value / 1000.0
    return value


def _revenue_cost_legend_label(component: str) -> str:
    labels = {
        "DA revenue": "DA revenue",
        "ID revenue": "ID revenue",
        "BCM capacity revenue": "BCM capacity revenue",
        "BCM-linked BEM activation revenue": "BCM-linked BEM activation revenue",
        "BEM activation revenue": "BEM activation revenue",
        "Degradation cost": "Degradation cost",
        "Auxiliary cost": "Auxiliary cost",
        "Transaction cost": "Transaction cost",
        "Activation penalty": "Activation penalty",
        "Terminal SoC repair": "Terminal SoC repair",
        "Total costs": "Total costs",
    }
    return labels.get(component, component)


def _revenue_cost_legend_label_wrapped(component: str) -> str:
    labels = {
        "BCM-linked BEM activation revenue": "BCM-linked BEM\nactivation revenue",
        "BCM capacity revenue": "BCM capacity\nrevenue",
        "BEM activation revenue": "BEM activation\nrevenue",
        "Terminal SoC repair": "Terminal SoC\nrepair",
    }
    return labels.get(component, _revenue_cost_legend_label(component))


def _revenue_cost_legend_label_tex(component: str) -> str:
    labels = {
        "BCM-linked BEM activation revenue": r"\shortstack[l]{BCM-linked BEM\\activation revenue}",
        "BCM capacity revenue": r"\shortstack[l]{BCM capacity\\revenue}",
        "BEM activation revenue": r"\shortstack[l]{BEM activation\\revenue}",
        "Terminal SoC repair": r"\shortstack[l]{Terminal SoC\\repair}",
    }
    return labels.get(component, _latex_escape(_revenue_cost_legend_label(component)))


def _choose_scale(values: list[float]) -> tuple[str, float]:
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not finite:
        return "EUR", 1.0
    mx = max(abs(v) for v in finite)
    if mx >= 50_000:
        return "kEUR", 1.0 / 1000.0
    return "EUR", 1.0


def _parse_formats(raw: str) -> set[str]:
    allowed = {"csv", "json", "png", "pdf", "tex"}
    formats = {item.strip().lower() for item in str(raw).split(",") if item.strip()}
    unknown = formats - allowed
    if unknown:
        raise ValueError(f"Unknown output format(s): {', '.join(sorted(unknown))}. Supported: {', '.join(sorted(allowed))}")
    if not formats:
        raise ValueError("At least one output format is required.")
    return formats


def _export_output_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Cannot export missing RQ3 output directory: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    shutil.copytree(source, destination)


def _prune_unselected_formats(root: Path, formats: set[str]) -> dict[str, int]:
    suffix_by_format = {
        "csv": ".csv",
        "json": ".json",
        "png": ".png",
        "pdf": ".pdf",
        "tex": ".tex",
    }
    prune_suffixes = {suffix for fmt, suffix in suffix_by_format.items() if fmt not in formats}
    counts = {suffix: 0 for suffix in sorted(prune_suffixes)}
    if not root.exists():
        return counts
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in prune_suffixes:
            continue
        path.unlink()
        counts[suffix] += 1
    return counts


def _read_rows_from_file(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Quantile-sweep manifests can wrap rows in a top-level list.
                row_list = next(
                    (data[key] for key in ("rows", "records", "scenarios", "summary") if isinstance(data.get(key), list)),
                    None,
                )
                df = pd.DataFrame(row_list) if row_list is not None else pd.DataFrame([data])
            else:
                return pd.DataFrame()
        else:
            df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read candidate RQ3 summary file {path}: {exc}") from exc
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        strategy_raw = row.get("trading_strategy", row.get("strategy", row.get("market_strategy", "")))
        strategy = normalize_strategy(strategy_raw, path)
        if strategy is None:
            strategy = _normalize_strategy_from_file(path)
        if strategy is None:
            continue
        annualized, value_col, method = _annualized_profit(row)
        if not math.isfinite(annualized):
            continue
        model_hint, quantile_hint = _extract_model_quantile_hint(path, row)
        model = str(row.get("model_key", row.get("model", row.get("model_name", model_hint or "")))).lower()
        quantile = _extract_quantile_tag(
            row.get("quantile_pair"),
            row.get("quantile_policy"),
            row.get("scenario"),
            row.get("quantile", row.get("quantile_low")),
            quantile_hint,
        )
        revenue_components: dict[str, tuple[float, str, str, bool]] = {}
        cost_components: dict[str, tuple[float, str, str, bool]] = {}

        for comp_name, candidates in REVENUE_COMPONENT_SPECS:
            comp_value = _resolve_component(row, candidates)
            revenue_components[comp_name] = comp_value

        for comp_name, candidates in COST_DETAIL_COMPONENT_SPECS:
            comp_value = _resolve_component(row, candidates)
            cost_components[comp_name] = comp_value

        total_cost = _resolve_component(row, COST_TOTAL_CANDIDATES)

        duration_days, duration_source = _duration_days(row)
        missing_duration_components = [
            comp_name for comp_name, (_, col, method, _) in {
                **revenue_components,
                **cost_components,
                "total_cost": total_cost,
            }.items()
            if method == "missing_duration"
        ]
        if missing_duration_components:
            raise ValueError(
                "Cannot annualize raw component(s) without a valid duration for decomposition rows. "
                f"File={path}, strategy={strategy}, model={model}, quantile={quantile}, "
                f"components={missing_duration_components}, duration_columns_tried=analysis_days,n_days,test_period_days,duration_days,start_utc_end_utc,start_end"
            )

        output_row = row.to_dict()
        output_row.update(
            {
                "strategy": strategy,
                "strategy_raw": strategy_raw,
                "annualized_net_profit_eur_per_year": annualized,
                "value_column_used": value_col,
                "annualization_method": method,
                "model": model,
                "quantile": quantile,
                "simulation_valid": row.get("simulation_valid", np.nan),
                "thesis_reportable": row.get("thesis_reportable", np.nan),
                "invalid_reason": row.get("invalid_reason", ""),
                "duration_days": duration_days,
                "duration_source": duration_source,
                "start_utc": row.get("start_utc", row.get("start", np.nan)),
                "end_utc": row.get("end_utc", row.get("end", np.nan)),
                "source_file": str(path),
                "revenue_components": revenue_components,
                "cost_components": cost_components,
                "total_cost_component": total_cost,
            }
        )
        rows.append(output_row)
    return pd.DataFrame(rows)


def _load_candidate(root: Path) -> Candidate | None:
    files = _candidate_files(root)
    if not files:
        return None
    frames = [_read_rows_from_file(path) for path in files]
    rows = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if rows.empty:
        return None
    detected = len(set(rows["strategy"].astype(str)) & set(STRATEGY_ORDER))
    xgb_p50 = rows["model"].astype(str).str.contains("xgb", case=False, na=False) & rows["quantile"].astype(str).str.contains("p50", case=False, na=False)
    mtime = max(path.stat().st_mtime for path in files)
    return Candidate(root=root, source_files=tuple(files), rows=rows, score=(detected, int(xgb_p50.sum()), float(mtime)))


def _load_cumulative_series_candidate(root: Path) -> tuple[pd.DataFrame, Path] | None:
    paths = _collect_ts_candidates(root)
    if not paths:
        return None
    required = set(STRATEGY_ORDER)
    best_partial: tuple[pd.DataFrame, Path] | None = None
    for priority in sorted({_ts_source_priority(path)[0] for path in paths}):
        frames: list[pd.DataFrame] = []
        used_sources: list[Path] = []
        for path in sorted(paths, key=_ts_source_priority):
            if _ts_source_priority(path)[0] != priority:
                continue
            try:
                frame = _read_ts_candidate(path)
            except ValueError:
                continue
            if frame.empty:
                continue
            frame["source_file"] = str(path)
            frames.append(frame)
            used_sources.append(path)
        if not frames:
            continue
        data = pd.concat(frames, ignore_index=True)
        data["strategy"] = data["strategy"].map(_canonicalize_strategy)
        data = data[data["strategy"].notna()].copy()
        if data.empty:
            continue
        xgb_p50 = (data["model"].astype(str).str.contains("xgb", case=False, na=False)) & (
            data["quantile"].astype(str).str.contains("p50", case=False, na=False)
        )
        if xgb_p50.any():
            data = data.loc[xgb_p50].copy()
        else:
            data["model"] = "xgb"
            data["quantile"] = "p50"
        data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True, errors="coerce")
        data = data.dropna(subset=["timestamp_utc", "strategy", "raw_pnl_eur"])
        data = data.loc[data["strategy"].isin(STRATEGY_ORDER)].copy()
        if data.empty:
            continue
        selected = data["source_file"].dropna()
        selected_path = Path(selected.iloc[0]) if not selected.empty else used_sources[0]
        detected = set(data["strategy"])
        if required.issubset(detected):
            return data, selected_path
        if best_partial is None or len(detected & required) > len(set(best_partial[0]["strategy"]) & required):
            best_partial = (data, selected_path)
    return best_partial


def _fallback_run_roots_for_missing_explicit(path: Path) -> list[Path]:
    """Return newest plausible run roots when a timestamped explicit path is stale.

    RQ3 simulation folders are timestamped, so thesis commands often become stale
    after a rerun. If the caller gives a missing timestamped folder, prefer the
    newest folder with the same run-name prefix instead of failing immediately.
    """
    parent = path.parent
    if not parent.exists():
        return []
    name = path.name
    patterns: list[str] = []
    timestamp_match = re.match(r"^(?P<prefix>.+)_\d{8}T\d{6}Z$", name)
    if timestamp_match:
        patterns.append(f"{timestamp_match.group('prefix')}_*")
    if name.startswith("rq3_xgb_p50_market_benchmark"):
        patterns.append("rq3_xgb_p50_market_benchmark_*")
    if not patterns:
        return []
    matches: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for candidate in parent.glob(pattern):
            if candidate.is_dir() and candidate not in seen:
                seen.add(candidate)
                matches.append(candidate)
    return sorted(matches, key=lambda p: (p.name, p.stat().st_mtime), reverse=True)


def discover_input(explicit_run_root: Path | None = None, search_roots: list[Path] | None = None) -> Candidate:
    roots: list[Path] = []
    if explicit_run_root is not None:
        if not explicit_run_root.exists():
            fallback_roots = _fallback_run_roots_for_missing_explicit(explicit_run_root)
            if not fallback_roots:
                raise FileNotFoundError(
                    f"Explicit RQ3 run root does not exist: {explicit_run_root}\n"
                    "Check the timestamped folder name on this machine, for example:\n"
                    "find artifacts/simulation_runs -maxdepth 1 -type d -name 'rq3_xgb_p50_market_benchmark_*' | sort"
                )
            print(
                "[WARN] Explicit RQ3 run root is missing; using newest matching run root instead: "
                f"{fallback_roots[0]}"
            )
            roots.append(fallback_roots[0])
        else:
            roots.append(explicit_run_root)
    else:
        for root in search_roots or DEFAULT_INPUT_ROOTS:
            if not root.exists():
                continue
            if root.name == "simulation_runs":
                roots.extend(sorted(root.glob("rq3_*"), key=lambda p: p.name, reverse=True))
            else:
                roots.append(root)
    candidates = [c for c in (_load_candidate(root) for root in roots) if c is not None]
    if not candidates:
        searched = "\n".join(f"  - {p}" for p in roots) or "  - <none>"
        raise FileNotFoundError(
            "No suitable RQ3 market-comparison input was found. Provide --run-root pointing to the completed "
            f"rq3_xgb_p50_market_benchmark_* folder. Expected files include performance_metrics_all_scenarios.csv, "
            f"performance_metrics.csv, backtest_summary.json, or quantile_sweep_summary.csv/json.\nSearched:\n{searched}"
        )
    candidates.sort(key=lambda c: c.score, reverse=True)
    best = candidates[0]
    if len(set(best.rows["strategy"]) & set(STRATEGY_ORDER)) < len(STRATEGY_ORDER):
        detected = sorted(set(best.rows["strategy"]))
        raise ValueError(
            "Selected RQ3 input does not contain all required strategies. "
            f"Required={STRATEGY_ORDER}; detected={detected}; root={best.root}"
        )
    return best


def select_strategy_rows(candidate: Candidate) -> pd.DataFrame:
    data = candidate.rows.copy()
    data = data.loc[data["strategy"].isin(STRATEGY_ORDER)].copy()
    if data.empty:
        raise ValueError(f"No required RQ3 strategies found in selected input: {candidate.root}")
    xgb_p50 = data["model"].astype(str).str.contains("xgb", case=False, na=False) & data["quantile"].astype(str).str.contains("p50", case=False, na=False)
    if xgb_p50.any():
        data = data.loc[xgb_p50].copy()
    data["_strategy_order"] = data["strategy"].map({name: idx for idx, name in enumerate(STRATEGY_ORDER)})
    def _source_rank(source_file: str) -> tuple[int, int]:
        source = str(source_file)
        strategy = _normalize_strategy_from_file(Path(source))
        preferred_folder = {
            "Multi": "/xgb_multi_p50/",
            "DA-only": "/xgb_da_p50/",
            "BCM-only": "/xgb_bcm_p50/",
            "BEM-only": "/xgb_bem_p50/",
        }.get(strategy, "")
        is_preferred_strategy_run = preferred_folder and preferred_folder in source
        is_other_xgb_strategy_run = "/xgb_" in source and "_p50/" in source
        return (
            0 if is_preferred_strategy_run else 1 if is_other_xgb_strategy_run else 2,
            0 if source.endswith("performance_metrics.csv") else 1,
        )

    data["_source_rank"] = data["source_file"].astype(str).map(_source_rank)
    data = data.sort_values(["_strategy_order", "_source_rank"]).drop_duplicates("strategy", keep="first")
    missing = [s for s in STRATEGY_ORDER if s not in set(data["strategy"])]
    if missing:
        raise ValueError(f"Missing required RQ3 strategies after filtering to XGB p50: {missing}")
    data = data.sort_values("_strategy_order").drop(columns=["_strategy_order", "_source_rank"])
    return data


def extract_decomposition_rows(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]], str, str]:
    """Build strategy-level annualized revenue/cost components.

    Returns:
      (component_frame, component_summary, scale, scale_label)
    """
    decomposition_rows: list[dict[str, Any]] = []
    component_status: dict[str, list[str]] = {"revenue": [], "cost": []}
    annualization_methods: set[str] = set()
    for _, row in data.iterrows():
        strategy = str(row["strategy"])
        row_out: dict[str, Any] = {"strategy": strategy}
        available_revenue_cols: list[str] = []
        available_cost_cols: list[str] = []
        # Revenue components
        for comp_name in REVENUE_COMPONENTS:
            value, col, method, is_annualized = row["revenue_components"][comp_name]
            annualization_methods.add(method)
            if not math.isfinite(value):
                is_structural = comp_name in STRUCTURALLY_ZERO_COMPONENTS.get(strategy, ())
                if is_structural:
                    component_status["revenue"].append(f"{strategy}: {comp_name} set to 0 (structural)")
                    value = 0.0
                else:
                    component_status["revenue"].append(f"{strategy}: {comp_name} unavailable (missing {col})")
                    value = math.nan
            row_out[f"revenue::{comp_name}"] = value
            if math.isfinite(value) and (value != 0 or is_annualized):
                available_revenue_cols.append(f"{comp_name}:{col}:{method}")

        detailed_cost_used = False
        for comp_name in COST_COMPONENTS:
            if comp_name == "Total costs":
                continue
            value, col, method, is_annualized = row["cost_components"][comp_name]
            annualization_methods.add(method)
            if math.isfinite(value):
                detailed_cost_used = True
                row_out[f"cost::{comp_name}"] = value
                available_cost_cols.append(f"{comp_name}:{col}:{method}")
            else:
                is_structural = False
                if strategy == "BCM-only" and comp_name in {"ID recourse cost", "Activation penalty", "Activation cost", "Auxiliary cost", "Degradation cost", "Transaction cost", "Offer cost", "Terminal SoC repair"}:
                    is_structural = True
                if is_structural:
                    component_status["cost"].append(f"{strategy}: {comp_name} set to 0 (structural)")
                    row_out[f"cost::{comp_name}"] = 0.0
                else:
                    component_status["cost"].append(f"{strategy}: {comp_name} unavailable (missing {col})")
                    row_out[f"cost::{comp_name}"] = math.nan

        if any(math.isfinite(v[0]) for v in row["cost_components"].values()):
            # If any detailed cost component available, omit total costs.
            row_out["cost::Total costs"] = 0.0
            row_out["cost_component_mode"] = "detailed"
        else:
            total_cost_value, total_col, total_method, _ = row["total_cost_component"]
            annualization_methods.add(total_method)
            if not math.isfinite(total_cost_value):
                total_cost_value = 0.0
                component_status["cost"].append(f"{strategy}: Total costs unavailable (missing {total_col})")
                row_out["cost_component_mode"] = "missing"
            else:
                row_out["cost_component_mode"] = "total"
            row_out["cost::Total costs"] = total_cost_value
            if total_cost_value != 0:
                available_cost_cols.append(f"Total costs:{total_col}:{total_method}")

        row_out["available_revenue_components"] = "; ".join(sorted(available_revenue_cols))
        row_out["available_cost_components"] = "; ".join(sorted(available_cost_cols))
        row_out["annualized_from"] = str(row["value_column_used"])
        row_out["annualization_method"] = row["annualization_method"]
        row_out["cost_component_mode"] = row_out["cost_component_mode"]
        row_out["duration_days"] = _safe_float(row["duration_days"])
        decomposition_rows.append(row_out)

    if not decomposition_rows:
        raise ValueError("No strategy rows available for decomposition after filtering.")
    decomposition = pd.DataFrame(decomposition_rows)
    # Determine unit based on finite components.
    all_values: list[float] = []
    for col in decomposition.columns:
        if col.startswith("revenue::") or col.startswith("cost::"):
            all_values.extend([float(v) for v in decomposition[col].tolist() if pd.notna(v) and math.isfinite(float(v))])
    unit, _ = _choose_scale(all_values)
    scale_label = f"Annualized {unit}/year"
    return decomposition, component_status, unit, scale_label


def plot_annualized_net_profit(data: pd.DataFrame, png_path: Path, pdf_path: Path, *, formats: set[str]) -> list[Path]:
    apply_geo_style()
    values_k = data["annualized_net_profit_eur_per_year"].to_numpy(dtype=float) / 1000.0
    labels = data["strategy"].tolist()
    colors = [STRATEGY_COLOR.get(label, THESIS_PALETTE["neutral_dark"]) for label in labels]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, values_k, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0.0, color=THESIS_PALETTE["neutral_dark"], linewidth=0.8)
    ax.set_xticks(x, labels=labels)
    ax.set_xlabel("")
    ax.set_ylabel("Annualized net profit (kEUR/year)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ymax = float(np.nanmax(values_k)) if len(values_k) else 0.0
    ymin = float(np.nanmin(values_k)) if len(values_k) else 0.0
    span = max(1.0, ymax - ymin)
    ax.set_ylim(min(0.0, ymin) - 0.12 * span, max(0.0, ymax) + 0.18 * span)
    for bar, value in zip(bars, values_k):
        offset = 0.035 * span
        va = "bottom" if value >= 0 else "top"
        y = value + offset if value >= 0 else value - offset
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:,.0f}", ha="center", va=va, fontsize=9)
    fig.subplots_adjust(bottom=0.32)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if "png" in formats:
        fig.savefig(png_path, dpi=220)
        paths.append(png_path)
    if "pdf" in formats:
        fig.savefig(pdf_path)
        paths.append(pdf_path)
    plt.close(fig)
    return paths


def write_latex_annualized_net_profit(data: pd.DataFrame, path: Path) -> Path:
    values_k = data["annualized_net_profit_eur_per_year"].to_numpy(dtype=float) / 1000.0
    labels = data["strategy"].tolist()
    ymax = float(np.nanmax(values_k)) if len(values_k) else 0.0
    ymin = float(np.nanmin(values_k)) if len(values_k) else 0.0
    span = max(1.0, ymax - ymin)
    y_min = min(0.0, ymin) - 0.12 * span
    y_max = max(0.0, ymax) + 0.18 * span
    color_names = {label: f"rqThree{label.replace('-', '').replace('only', 'Only')}" for label in labels}
    nodes = []
    bars = []
    for idx, (label, value) in enumerate(zip(labels, values_k)):
        y0 = min(0.0, float(value))
        y1 = max(0.0, float(value))
        bars.append(
            rf"\filldraw[fill={color_names[label]}, draw=white, line width=0.6pt] "
            rf"(axis cs:{_tex_float(idx - 0.32, 3)},{_tex_float(y0, 4)}) rectangle "
            rf"(axis cs:{_tex_float(idx + 0.32, 3)},{_tex_float(y1, 4)});"
        )
        offset = 0.035 * span
        y = value + offset if value >= 0 else value - offset
        anchor = "south" if value >= 0 else "north"
        nodes.append(rf"\node[font=\small, anchor={anchor}] at (axis cs:{idx},{_tex_float(y, 4)}) {{{_latex_escape(f'{value:,.0f}')}}};")
    lines = [
        r"% Requires \usepackage{pgfplots}",
        r"% Requires \pgfplotsset{compat=1.18}",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        _tex_color_def("rqThreeNeutral", THESIS_PALETTE["neutral_dark"]),
        _tex_color_def("rqThreeGrid", "#D8D8D8"),
        *[_tex_color_def(color_names[label], STRATEGY_COLOR.get(label, THESIS_PALETTE["neutral_dark"])) for label in labels],
        r"\begin{axis}[",
        r"tick align=outside,",
        r"axis line style={rqThreeNeutral},",
        r"tick style={rqThreeNeutral},",
        r"label style={font=\normalsize},",
        r"tick label style={font=\normalsize},",
        r"grid=major,",
        r"grid style={rqThreeGrid!55, line width=0.2pt},",
        r"width=0.78\linewidth,",
        r"height=0.48\linewidth,",
        r"xlabel={Market strategy},",
        r"ylabel={Annualized net profit (kEUR/year)},",
        "xmin=-0.5, xmax=" + _tex_float(len(labels) - 0.5, 1) + ",",
        "xtick={" + ",".join(str(i) for i in range(len(labels))) + "},",
        "xticklabels={" + ",".join(_latex_escape(label) for label in labels) + "},",
        rf"ymin={_tex_float(y_min, 4)}, ymax={_tex_float(y_max, 4)},",
        r"axis y line*=left,",
        r"axis x line*=bottom,",
        r"]",
        *bars,
    ]
    lines.extend(
        [
            rf"\draw[rqThreeNeutral, line width=0.7pt] (axis cs:-0.5,0) -- (axis cs:{_tex_float(len(labels) - 0.5, 1)},0);",
            *nodes,
            r"\end{axis}",
            r"\end{tikzpicture}",
            r"\caption{Annualized net profit by market participation strategy. The figure compares the XGB p50 multi-market strategy with DA-only, BCM-only and BEM-only baselines to evaluate whether revenue stacking improves BESS profitability.}",
            r"\label{fig:rq3-annualized-net-profit-by-market-strategy}",
            r"\end{figure}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_cleared_bid_volume_market_comparison(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        raise ValueError("Cannot build RQ3 cleared bid-volume comparison from empty strategy data.")
    by_strategy = {str(row["strategy"]): row for _, row in data.iterrows()}
    missing = [strategy for strategy in ["Multi", *RQ3_SINGLE_MARKET_STRATEGY.values()] if strategy not in by_strategy]
    if missing:
        raise ValueError(f"Missing strategy rows for RQ3 cleared bid-volume comparison: {', '.join(missing)}")

    def _metric(strategy: str, metric: str) -> float:
        value = _safe_float(by_strategy[strategy].get(metric, math.nan))
        if not math.isfinite(value):
            raise ValueError(f"Missing or invalid {metric!r} for {strategy} in RQ3 cleared bid-volume comparison.")
        return value

    def _annualization(strategy: str) -> float:
        factor = _safe_float(by_strategy[strategy].get("annualization_factor", math.nan))
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError(f"Missing or invalid annualization_factor for {strategy} in RQ3 cleared bid-volume comparison.")
        return factor

    def _executed_ledger_volume(strategy: str, columns: tuple[str, ...]) -> float:
        source_file = str(by_strategy[strategy].get("source_file", "") or "")
        if not source_file:
            raise ValueError(f"Missing source_file for {strategy} in RQ3 cleared bid-volume comparison.")
        ledger_path = Path(source_file).parent / "executed_ledger.parquet"
        if not ledger_path.exists():
            raise FileNotFoundError(f"Missing executed ledger for {strategy} cleared bid-volume comparison: {ledger_path}")
        ledger = pd.read_parquet(ledger_path)
        missing_cols = [col for col in columns if col not in ledger.columns]
        if missing_cols:
            raise ValueError(f"Missing cleared volume columns for {strategy}: {', '.join(missing_cols)} in {ledger_path}")
        total = 0.0
        for col in columns:
            values = pd.to_numeric(ledger[col], errors="coerce").fillna(0.0)
            total += float(values.abs().sum())
        return total

    rows: list[dict[str, Any]] = []
    for market in RQ3_MARKET_ORDER:
        single_strategy = RQ3_SINGLE_MARKET_STRATEGY[market]
        if market == "DA":
            metric = "da_realized_abs_mwh_total"
            unit = "MWh"
        elif market == "BCM":
            metric = "executed_ledger:real_executed_bcm_capacity_pos_mw+real_executed_bcm_capacity_neg_mw"
            unit = "MW-h"
        else:
            metric = "bem_realized_abs_mwh_total"
            unit = "MWh"
        for strategy_group, strategy in (("Single-market strategy", single_strategy), ("Multi-market strategy", "Multi")):
            if market == "BCM":
                raw_value = _executed_ledger_volume(
                    strategy,
                    ("real_executed_bcm_capacity_pos_mw", "real_executed_bcm_capacity_neg_mw"),
                )
            else:
                raw_value = _metric(strategy, metric)
            annualized = raw_value * _annualization(strategy)
            rows.append(
                {
                    "market": market,
                    "strategy_group": strategy_group,
                    "strategy": strategy,
                    "source_metric": metric,
                    "cleared_volume_total": raw_value,
                    "cleared_volume_unit": unit,
                    "annualization_factor": _annualization(strategy),
                    "cleared_volume_annualized_mwh_equivalent": annualized,
                    "cleared_volume_annualized_gwh_equivalent": annualized / 1000.0,
                }
            )
    out = pd.DataFrame(rows)
    out["_market_order"] = out["market"].map({market: idx for idx, market in enumerate(RQ3_MARKET_ORDER)})
    out["_group_order"] = out["strategy_group"].map({"Single-market strategy": 0, "Multi-market strategy": 1})
    return out.sort_values(["_market_order", "_group_order"]).drop(columns=["_market_order", "_group_order"])


def plot_cleared_bid_volume_market_comparison(data: pd.DataFrame, png_path: Path, pdf_path: Path, *, formats: set[str]) -> list[Path]:
    apply_geo_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    x = np.arange(len(RQ3_MARKET_ORDER), dtype=float)
    width = 0.34
    groups = ["Single-market strategy", "Multi-market strategy"]
    offsets = {"Single-market strategy": -width / 2, "Multi-market strategy": width / 2}
    hatches = {"Single-market strategy": "", "Multi-market strategy": "//"}
    max_value = 0.0
    for group in groups:
        values = []
        colors = []
        for market in RQ3_MARKET_ORDER:
            match = data.loc[data["market"].eq(market) & data["strategy_group"].eq(group)]
            value = _safe_float(match["cleared_volume_annualized_gwh_equivalent"].iloc[0]) if not match.empty else 0.0
            values.append(value if math.isfinite(value) else 0.0)
            colors.append(RQ3_MARKET_COLOR.get(market, THESIS_PALETTE["neutral_dark"]))
        max_value = max(max_value, max(values) if values else 0.0)
        bars = ax.bar(
            x + offsets[group],
            values,
            width=width,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            hatch=hatches[group],
            label=group,
        )
        for bar, value in zip(bars, values):
            if value <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + max(0.02 * max_value, 0.03),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )
    ax.set_xticks(x, labels=RQ3_MARKET_ORDER)
    ax.set_xlabel("Market")
    ax.set_ylabel("Annualized cleared bid volume (GWh/year)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_ylim(0.0, max(1.0, max_value * 1.22))
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if "png" in formats:
        fig.savefig(png_path, dpi=220)
        paths.append(png_path)
    if "pdf" in formats:
        fig.savefig(pdf_path)
        paths.append(pdf_path)
    plt.close(fig)
    return paths


def write_latex_cleared_bid_volume_market_comparison(data: pd.DataFrame, path: Path) -> Path:
    groups = ["Single-market strategy", "Multi-market strategy"]
    group_colors = {"Single-market strategy": "rqThreeSingleBid", "Multi-market strategy": "rqThreeMultiBid"}
    x_by_market = {market: idx for idx, market in enumerate(RQ3_MARKET_ORDER)}
    width = 0.32
    offsets = {"Single-market strategy": -0.18, "Multi-market strategy": 0.18}
    max_value = max(1.0, float(pd.to_numeric(data["cleared_volume_annualized_gwh_equivalent"], errors="coerce").max()))
    y_max = max_value * 1.24
    bars: list[str] = []
    labels: list[str] = []
    for group in groups:
        for market in RQ3_MARKET_ORDER:
            match = data.loc[data["market"].eq(market) & data["strategy_group"].eq(group)]
            value = _safe_float(match["cleared_volume_annualized_gwh_equivalent"].iloc[0]) if not match.empty else 0.0
            if not math.isfinite(value):
                value = 0.0
            x = x_by_market[market] + offsets[group]
            color = group_colors[group]
            bars.append(
                rf"\filldraw[fill={color}, draw=white, line width=0.5pt] "
                rf"(axis cs:{_tex_float(x - width / 2, 3)},0) rectangle "
                rf"(axis cs:{_tex_float(x + width / 2, 3)},{_tex_float(value, 4)});"
            )
            if value > 0:
                label_offset = max(0.08, min(0.55, value * 0.08))
                labels.append(
                    rf"\node[font=\small, anchor=south] at (axis cs:{_tex_float(x, 3)},{_tex_float(value + label_offset, 4)}) {{{_tex_float(value, 1)}}};"
                )
    lines = [
        r"% Requires \usepackage{pgfplots}",
        r"% Requires \pgfplotsset{compat=1.18}",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        _tex_color_def("rqThreeNeutral", THESIS_PALETTE["neutral_dark"]),
        _tex_color_def("rqThreeGrid", "#D8D8D8"),
        _tex_color_def("rqThreeSingleBid", THESIS_PALETTE["neutral_dark"]),
        _tex_color_def("rqThreeMultiBid", THESIS_PALETTE["primary"]),
        r"\begin{axis}[",
        r"tick align=outside,",
        r"axis line style={rqThreeNeutral},",
        r"tick style={rqThreeNeutral},",
        r"label style={font=\small},",
        r"tick label style={font=\small},",
        r"grid=major,",
        r"grid style={rqThreeGrid!55, line width=0.2pt},",
        r"width=0.82\linewidth,",
        r"height=0.46\linewidth,",
        r"xlabel={Market},",
        r"ylabel={Annualized cleared bid volume (GWh/year)},",
        "xmin=-0.6, xmax=" + _tex_float(len(RQ3_MARKET_ORDER) - 0.4, 1) + ",",
        "xtick={" + ",".join(str(i) for i in range(len(RQ3_MARKET_ORDER))) + "},",
        "xticklabels={" + ",".join(RQ3_MARKET_ORDER) + "},",
        rf"ymin=0, ymax={_tex_float(y_max, 4)},",
        r"axis y line*=left,",
        r"axis x line*=bottom,",
        r"legend columns=2,",
        r"legend cell align=left,",
        r"legend style={at={(0.5,-0.26)}, anchor=north, font=\small, draw=none, fill=none, /tikz/every even column/.append style={column sep=0.45cm}},",
        r"]",
        *bars,
        *labels,
        r"\addlegendimage{area legend, fill=rqThreeSingleBid, draw=white}",
        r"\addlegendentry{Single-market strategy}",
        r"\addlegendimage{area legend, fill=rqThreeMultiBid, draw=white}",
        r"\addlegendentry{Multi-market strategy}",
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\caption{Annualized cleared bid volume by market, comparing each single-market strategy with the corresponding market volume inside the XGB p50 multi-market strategy.}",
        r"\label{fig:rq3-cleared-bid-volume-market-comparison}",
        r"\end{figure}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_cleared_bid_volume_market_comparison(data: pd.DataFrame, out_root: Path, *, formats: set[str]) -> list[Path]:
    comparison = build_cleared_bid_volume_market_comparison(data)
    csv_dir = out_root / "result_section" / "csv"
    figures_dir = out_root / "result_section" / "figures"
    latex_dir = out_root / "result_section" / "latex_figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "cleared_bid_volume_market_comparison.csv"
    png_path = figures_dir / "cleared_bid_volume_market_comparison.png"
    pdf_path = figures_dir / "cleared_bid_volume_market_comparison.pdf"
    tex_path = latex_dir / "cleared_bid_volume_market_comparison.tex"
    outputs: list[Path] = []
    if "csv" in formats:
        comparison.to_csv(csv_path, index=False)
        outputs.append(csv_path)
    outputs.extend(plot_cleared_bid_volume_market_comparison(comparison, png_path, pdf_path, formats=formats))
    if "tex" in formats:
        outputs.append(write_latex_cleared_bid_volume_market_comparison(comparison, tex_path))
    return outputs


def build_bcm_revenue_mechanism_comparison(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        raise ValueError("Cannot build BCM revenue mechanism comparison from empty strategy data.")
    by_strategy = {str(row["strategy"]): row for _, row in data.iterrows()}
    required = ["BCM-only", "Multi"]
    missing = [strategy for strategy in required if strategy not in by_strategy]
    if missing:
        raise ValueError(f"Missing strategy rows for BCM revenue mechanism comparison: {', '.join(missing)}")

    def _annualization(row: pd.Series) -> float:
        factor = _safe_float(row.get("annualization_factor", math.nan))
        if math.isfinite(factor) and factor > 0:
            return factor
        days = _safe_float(row.get("duration_days", math.nan))
        if math.isfinite(days) and days > 0:
            return 365.0 / days
        raise ValueError(f"Missing annualization factor for {row.get('strategy', '<unknown>')} in BCM revenue mechanism comparison.")

    def _source_dir(row: pd.Series) -> Path:
        source_file = str(row.get("source_file", "") or "")
        if not source_file:
            raise ValueError(f"Missing source_file for {row.get('strategy', '<unknown>')} in BCM revenue mechanism comparison.")
        return Path(source_file).parent

    def _read_planned(row: pd.Series) -> pd.DataFrame:
        path = _source_dir(row) / "planned_ledger.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing planned ledger for BCM revenue mechanism comparison: {path}")
        return pd.read_parquet(path)

    def _read_executed(row: pd.Series) -> pd.DataFrame:
        path = _source_dir(row) / "executed_ledger.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing executed ledger for BCM revenue mechanism comparison: {path}")
        return pd.read_parquet(path)

    def _planned_num(df: pd.DataFrame, col: str) -> pd.Series:
        if col not in df.columns:
            raise ValueError(f"Missing required BCM planned-ledger column: {col}")
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    def _executed_num(df: pd.DataFrame, col: str) -> pd.Series:
        if col not in df.columns:
            raise ValueError(f"Missing required BCM executed-ledger column: {col}")
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    rows: list[dict[str, Any]] = []
    for strategy in required:
        row = by_strategy[strategy]
        planned = _read_planned(row)
        executed = _read_executed(row)
        ann = _annualization(row)
        cand_pos = _planned_num(planned, "bcm_precommit_candidate_pos_mw").clip(lower=0.0)
        cand_neg = _planned_num(planned, "bcm_precommit_candidate_neg_mw").clip(lower=0.0)
        locked_pos = _planned_num(planned, "bcm_precommit_locked_pos_mw").clip(lower=0.0)
        locked_neg = _planned_num(planned, "bcm_precommit_locked_neg_mw").clip(lower=0.0)
        cand_abs = cand_pos + cand_neg
        locked_abs = locked_pos + locked_neg
        candidate_mask = cand_abs > 1e-9
        fully_blocked_mask = candidate_mask & (locked_abs <= 1e-9)
        partially_reduced_mask = (cand_abs - locked_abs > 1e-9) & (locked_abs > 1e-9)
        executed_abs = (
            _executed_num(executed, "real_executed_bcm_capacity_pos_mw").abs()
            + _executed_num(executed, "real_executed_bcm_capacity_neg_mw").abs()
        )
        capacity_revenue = _safe_float(row.get("annualized_bcm_capacity_revenue_eur", math.nan))
        activation_revenue = _safe_float(row.get("annualized_bcm_activation_revenue_eur", math.nan))
        if not math.isfinite(capacity_revenue):
            capacity_revenue = _safe_float(row.get("bcm_capacity_revenue_eur", math.nan)) * ann
        if not math.isfinite(activation_revenue):
            activation_revenue = _safe_float(row.get("bcm_linked_activation_revenue_eur", row.get("bcm_activation_revenue_eur", math.nan))) * ann
        total_revenue = capacity_revenue + activation_revenue
        locked_total = float(locked_abs.sum())
        rows.append(
            {
                "strategy": strategy,
                "bcm_candidate_hours": int(candidate_mask.sum()),
                "bcm_fully_blocked_candidate_hours": int(fully_blocked_mask.sum()),
                "bcm_partially_reduced_candidate_hours": int(partially_reduced_mask.sum()),
                "bcm_locked_hours": int((locked_abs > 1e-9).sum()),
                "bcm_candidate_volume_mw_h": float(cand_abs.sum()),
                "bcm_blocked_candidate_volume_mw_h": float(cand_abs[fully_blocked_mask].sum()),
                "bcm_locked_volume_mw_h": locked_total,
                "bcm_executed_capacity_volume_mw_h": float(executed_abs.sum()),
                "bcm_candidate_to_locked_share": locked_total / float(cand_abs.sum()) if float(cand_abs.sum()) > 0 else math.nan,
                "bcm_blocked_candidate_hour_share": float(fully_blocked_mask.sum()) / float(candidate_mask.sum()) if int(candidate_mask.sum()) else math.nan,
                "bcm_annualized_capacity_revenue_eur": capacity_revenue,
                "bcm_annualized_linked_activation_revenue_eur": activation_revenue,
                "bcm_annualized_total_revenue_eur": total_revenue,
                "bcm_revenue_per_locked_mw_h_eur": total_revenue / (locked_total * ann) if locked_total > 0 and ann > 0 else math.nan,
                "annualization_factor": ann,
            }
        )
    out = pd.DataFrame(rows)
    out["_strategy_order"] = out["strategy"].map({"BCM-only": 0, "Multi": 1})
    return out.sort_values("_strategy_order").drop(columns="_strategy_order")


def _fmt_table_num(value: Any, digits: int = 0) -> str:
    x = _safe_float(value)
    if not math.isfinite(x):
        return "--"
    return f"{x:,.{digits}f}"


def _fmt_table_pct(value: Any, digits: int = 0) -> str:
    x = _safe_float(value)
    if not math.isfinite(x):
        return "--"
    return f"{100.0 * x:.{digits}f}\\%"


def write_latex_bcm_revenue_mechanism_comparison(data: pd.DataFrame, path: Path) -> Path:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{BCM revenue mechanism comparison for BCM-only and the BCM component of the multi-market strategy.}",
        r"\label{tab:rq3-bcm-revenue-mechanism-comparison}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        (
            r"Strategy & Candidate h & Blocked h & Blocked share & Locked h & "
            r"Candidate MW-h & Locked MW-h & BCM revenue (kEUR/y) & Revenue per locked MW-h \\"
        ),
        r"\midrule",
    ]
    for _, row in data.iterrows():
        cells = [
            _latex_escape(row["strategy"]),
            _fmt_table_num(row["bcm_candidate_hours"], 0),
            _fmt_table_num(row["bcm_fully_blocked_candidate_hours"], 0),
            _fmt_table_pct(row["bcm_blocked_candidate_hour_share"], 0),
            _fmt_table_num(row["bcm_locked_hours"], 0),
            _fmt_table_num(row["bcm_candidate_volume_mw_h"], 0),
            _fmt_table_num(row["bcm_locked_volume_mw_h"], 0),
            _fmt_table_num(row["bcm_annualized_total_revenue_eur"] / 1000.0, 0),
            _fmt_table_num(row["bcm_revenue_per_locked_mw_h_eur"], 0),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            (
                r"\par\vspace{0.3em}\footnotesize\raggedright "
                r"Candidate and locked volumes are test-period MW-h; revenue is annualized. "
                r"BCM revenue combines BCM capacity revenue and BCM-linked BEM activation revenue. "
            ),
            r"\end{table}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_bcm_revenue_mechanism_comparison(data: pd.DataFrame, out_root: Path, *, formats: set[str]) -> list[Path]:
    comparison = build_bcm_revenue_mechanism_comparison(data)
    csv_dir = out_root / "result_section" / "csv"
    latex_dir = out_root / "result_section" / "latex_tables"
    csv_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "bcm_revenue_mechanism_comparison.csv"
    tex_path = latex_dir / "bcm_revenue_mechanism_comparison.tex"
    outputs: list[Path] = []
    if "csv" in formats:
        comparison.to_csv(csv_path, index=False)
        outputs.append(csv_path)
    if "tex" in formats:
        outputs.append(write_latex_bcm_revenue_mechanism_comparison(comparison, tex_path))
    return outputs


def _revenue_components_for_plot(decomp: pd.DataFrame) -> list[str]:
    out: list[str] = []
    available = decomp.get("available_revenue_components", pd.Series(dtype=str)).astype(str)
    for name in REVENUE_COMPONENTS:
        col = f"revenue::{name}"
        if col not in decomp.columns:
            continue
        values = pd.to_numeric(decomp[col], errors="coerce")
        if values.abs().gt(1e-12).any() or available.str.contains(name, regex=False).any():
            out.append(name)
    return out


def _cost_components_for_plot(decomp: pd.DataFrame) -> list[str]:
    detailed_components = [name for name in COST_COMPONENTS if name != "Total costs" and f"cost::{name}" in decomp.columns]
    if decomp.empty:
        return []
    if all(decomp.get("cost_component_mode", "") == "total") and "cost::Total costs" in decomp.columns:
        return ["Total costs"]
    available = decomp.get("available_cost_components", pd.Series(dtype=str)).astype(str)
    out: list[str] = []
    for name in detailed_components:
        values = pd.to_numeric(decomp[f"cost::{name}"], errors="coerce")
        if values.abs().gt(1e-12).any() or available.str.contains(name, regex=False).any():
            out.append(name)
    if out:
        return out
    if "cost::Total costs" in decomp.columns:
        return ["Total costs"]
    return out


def _decomposition_long(decomp: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    revenue_components = _revenue_components_for_plot(decomp)
    cost_components = _cost_components_for_plot(decomp)
    for _, row in decomp.iterrows():
        strategy = str(row["strategy"])
        for name in revenue_components:
            col = f"revenue::{name}"
            rows.append(
                {
                    "strategy": strategy,
                    "component_type": "revenue",
                    "component": name,
                    "annualized_component_eur_per_year": _safe_float(row[col]),
                    "cost_component_mode": row.get("cost_component_mode", ""),
                    "available_revenue_components": row.get("available_revenue_components", ""),
                    "available_cost_components": row.get("available_cost_components", ""),
                    "annualization_method": row.get("annualization_method", ""),
                    "source_duration_days": row.get("duration_days", np.nan),
                }
            )
        for name in cost_components:
            col = f"cost::{name}"
            rows.append(
                {
                    "strategy": strategy,
                    "component_type": "cost",
                    "component": name,
                    "annualized_component_eur_per_year": -abs(_safe_float(row[col])),
                    "cost_component_mode": row.get("cost_component_mode", ""),
                    "available_revenue_components": row.get("available_revenue_components", ""),
                    "available_cost_components": row.get("available_cost_components", ""),
                    "annualization_method": row.get("annualization_method", ""),
                    "source_duration_days": row.get("duration_days", np.nan),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_strategy_order"] = out["strategy"].map({name: idx for idx, name in enumerate(STRATEGY_ORDER)})
    component_order = {name: idx for idx, name in enumerate(list(REVENUE_COMPONENTS) + list(COST_COMPONENTS))}
    out["_component_order"] = out["component"].map(component_order)
    out["_component_type_order"] = out["component_type"].map({"revenue": 0, "cost": 1})
    return out.sort_values(["_strategy_order", "_component_type_order", "_component_order"]).drop(
        columns=["_strategy_order", "_component_type_order", "_component_order"]
    )


def _plot_revenue_cost_decomposition(decomp: pd.DataFrame, unit: str, unit_scale: float, out_path: Path, pdf_path: Path, *, formats: set[str]) -> list[Path]:
    apply_geo_style()
    strategies = decomp["strategy"].tolist()
    x = np.arange(len(strategies))
    revenue_components = _revenue_components_for_plot(decomp)
    cost_components = _cost_components_for_plot(decomp)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    rev_bottom = np.zeros(len(strategies), dtype=float)
    cost_bottom = np.zeros(len(strategies), dtype=float)
    for name in revenue_components:
        y = np.nan_to_num(decomp[f"revenue::{name}"].to_numpy(dtype=float) * unit_scale, nan=0.0)
        bottom = np.where(y >= 0.0, rev_bottom, cost_bottom)
        color = REVENUE_COMPONENT_COLOR.get(name, THESIS_PALETTE["primary"])
        ax.bar(
            x,
            y,
            bottom=bottom,
            width=0.78,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
        for idx, (value, base) in enumerate(zip(y, bottom)):
            if value <= 0.0 or value < 18.0:
                continue
            x_offset, y_offset, ha, _ = _revenue_label_position(strategies[idx], name)
            text_color = _revenue_label_text_color(name, "west" if ha == "left" else "center")
            ax.text(
                x[idx] + x_offset,
                base + value / 2.0 + y_offset,
                _revenue_segment_label(value),
                ha=ha,
                va="center",
                fontsize=8.5,
                color=text_color,
                fontweight="normal",
                zorder=20,
                clip_on=False,
            )
        rev_bottom = rev_bottom + np.maximum(y, 0.0)
        cost_bottom = cost_bottom + np.minimum(y, 0.0)

    for name in cost_components:
        raw = decomp[f"cost::{name}"].to_numpy(dtype=float) * unit_scale
        y = -np.abs(np.nan_to_num(raw, nan=0.0))
        ax.bar(
            x,
            y,
            bottom=cost_bottom,
            width=0.78,
            color=COST_COMPONENT_COLOR.get(name, THESIS_PALETTE["neutral_dark"]),
            edgecolor="white",
            linewidth=0.6,
        )
        cost_bottom = cost_bottom + y

    ymin = min(0.0, float(np.nanmin(cost_bottom))) if len(cost_bottom) else -1.0
    ymax = max(0.0, float(np.nanmax(rev_bottom))) if len(rev_bottom) else 1.0
    span = max(1.0, ymax - ymin)

    ax.axhline(0.0, color=THESIS_PALETTE["neutral_dark"], linewidth=0.8)
    ax.set_xlim(-0.5, len(strategies) - 0.5)
    ax.set_xticks(x, labels=[_revenue_cost_strategy_axis_label(strategy) for strategy in strategies])
    ax.set_xlabel("Market strategy")
    ax.set_ylabel(f"Annualized value ({unit}/year)")
    ax.set_ylim(ymin - 0.16 * span, ymax + 0.16 * span)

    if revenue_components or cost_components:
        empty_handle = Patch(facecolor="none", edgecolor="none", alpha=0.0)
        handles: list[Patch] = [empty_handle, empty_handle, empty_handle, empty_handle]
        labels: list[str] = [REVENUE_LABEL, "", COST_LABEL, ""]
        legend_rows = max(math.ceil(len(revenue_components) / 2), math.ceil(len(cost_components) / 2))
        for row_idx in range(legend_rows):
            for components, color_map in (
                (revenue_components, REVENUE_COMPONENT_COLOR),
                (cost_components, COST_COMPONENT_COLOR),
            ):
                row_components = components[2 * row_idx : 2 * row_idx + 2]
                for name in row_components:
                    handles.append(Patch(facecolor=color_map.get(name, THESIS_PALETTE["neutral_dark"]), edgecolor="white"))
                    labels.append(_revenue_cost_legend_label_wrapped(name))
                for _ in range(2 - len(row_components)):
                    handles.append(empty_handle)
                    labels.append("")
        ax.legend(
            handles=handles,
            labels=labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.34),
            ncol=4,
            frameon=False,
            fontsize=7,
            alignment="left",
            columnspacing=1.0,
            handletextpad=0.45,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if "png" in formats:
        fig.savefig(out_path, dpi=220)
        paths.append(out_path)
    if "pdf" in formats:
        fig.savefig(pdf_path)
        paths.append(pdf_path)
    plt.close(fig)
    return paths


def write_latex_revenue_cost_decomposition(decomp: pd.DataFrame, unit: str, unit_scale: float, path: Path) -> Path:
    strategies = decomp["strategy"].tolist()
    revenue_components = _revenue_components_for_plot(decomp)
    cost_components = _cost_components_for_plot(decomp)

    rev_color_names: dict[str, str] = {
        name: f"rqThreeRev{''.join(ch for ch in name if ch.isalnum())}" for name in revenue_components
    }
    cost_color_names: dict[str, str] = {
        name: f"rqThreeCost{''.join(ch for ch in name if ch.isalnum())}" for name in cost_components
    }

    lines: list[str] = [
        r"% Requires \usepackage{pgfplots}",
        r"% Requires \pgfplotsset{compat=1.18}",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
    ]
    for name, color_name in rev_color_names.items():
        lines.append(_tex_color_def(color_name, REVENUE_COMPONENT_COLOR.get(name, THESIS_PALETTE["primary"])))
    for name, color_name in cost_color_names.items():
        lines.append(_tex_color_def(color_name, COST_COMPONENT_COLOR.get(name, THESIS_PALETTE["neutral_dark"])))

    lines.extend(
        [
            r"\begin{axis}[",
            r"ybar stacked,",
            r"bar width=18pt,",
            r"width=0.88\linewidth,",
            r"height=0.44\linewidth,",
            r"clip=false,",
            r"tick align=outside,",
            r"axis line style={black},",
            r"tick style={black},",
            r"enlarge x limits=false,",
            r"xmin=-0.5,",
            rf"xmax={_tex_float(len(strategies) - 0.5, 1)},",
            f"xtick={{{','.join(str(i) for i in range(len(strategies)))}}},",
            f"xticklabels={{{','.join(_latex_escape(_revenue_cost_strategy_axis_label(s)) for s in strategies)}}},",
            f"ylabel={{Annualized value ({_latex_escape(unit)}/year)}},",
            r"legend cell align=left,",
            r"legend style={at={(0.5,-0.24)}, anchor=north, font=\scriptsize, draw=none, fill=none, /tikz/every even column/.append style={column sep=0.45cm}},",
            r"legend columns=4,",
            r"]",
        ]
    )

    for name in revenue_components:
        vals = np.nan_to_num(decomp[f"revenue::{name}"].to_numpy(dtype=float) * unit_scale, nan=0.0)
        coords = " ".join(f"({idx},{_tex_float(float(v), 4)})" for idx, v in enumerate(vals))
        color = rev_color_names[name]
        lines.append(rf"\addplot+[draw=white, fill={color}, forget plot] coordinates {{{coords}}};")

    for name in cost_components:
        vals = -np.abs(np.nan_to_num(decomp[f"cost::{name}"].to_numpy(dtype=float), nan=0.0) * unit_scale)
        coords = " ".join(f"({idx},{_tex_float(float(v), 4)})" for idx, v in enumerate(vals))
        color = cost_color_names[name]
        lines.append(rf"\addplot+[draw=white, fill={color}, forget plot] coordinates {{{coords}}};")

    lines.extend(
        [
            r"\addlegendimage{empty legend}",
            rf"\addlegendentry{{{REVENUE_LABEL}}}",
            r"\addlegendimage{empty legend}",
            r"\addlegendentry{}",
            r"\addlegendimage{empty legend}",
            rf"\addlegendentry{{{COST_LABEL}}}",
            r"\addlegendimage{empty legend}",
            r"\addlegendentry{}",
        ]
    )
    legend_rows = max(math.ceil(len(revenue_components) / 2), math.ceil(len(cost_components) / 2))
    for row_idx in range(legend_rows):
        for components, color_names in (
            (revenue_components, rev_color_names),
            (cost_components, cost_color_names),
        ):
            row_components = components[2 * row_idx : 2 * row_idx + 2]
            for component in row_components:
                lines.extend(
                    [
                        rf"\addlegendimage{{area legend, fill={color_names[component]}, draw=white}}",
                        rf"\addlegendentry{{{_revenue_cost_legend_label_tex(component)}}}",
                    ]
                )
            for _ in range(2 - len(row_components)):
                lines.extend(
                    [
                        r"\addlegendimage{empty legend}",
                        r"\addlegendentry{}",
                    ]
                )
    label_nodes: list[str] = []
    label_counter = 0
    rev_positive_bottom = np.zeros(len(strategies), dtype=float)
    for name in revenue_components:
        vals = np.nan_to_num(decomp[f"revenue::{name}"].to_numpy(dtype=float) * unit_scale, nan=0.0)
        for idx, value in enumerate(vals):
            if value <= 0.0 or value < 18.0:
                continue
            x_offset, y_offset, _, anchor = _revenue_label_position(strategies[idx], name)
            tex_x_offset = 0.0 if anchor == "west" else x_offset
            tex_xshift = _revenue_label_tex_xshift(anchor)
            y_mid = rev_positive_bottom[idx] + float(value) / 2.0 + y_offset
            text_color = _revenue_label_text_color(name, anchor)
            coord_name = f"rqThreeRevenueLabel{label_counter}"
            label_counter += 1
            lines.append(
                rf"\coordinate ({coord_name}) at "
                rf"(axis cs:{_tex_float(idx + tex_x_offset, 4)},{_tex_float(y_mid, 4)});"
            )
            label_nodes.append(
                rf"\node[font=\small, text={text_color}, anchor={anchor}, inner sep=0pt, xshift={tex_xshift}] "
                rf"at ({coord_name}) {{{_latex_escape(_revenue_segment_label(float(value)))}}};"
            )
        rev_positive_bottom = rev_positive_bottom + np.maximum(vals, 0.0)
    if strategies:
        lines.append(rf"\draw[black, line width=0.45pt] (axis cs:-0.5,0) -- (axis cs:{len(strategies) - 0.5},0);")
    lines.extend(
        [
            r"\end{axis}",
            *label_nodes,
            r"\end{tikzpicture}",
            r"\caption{Revenue and cost decomposition by market participation strategy. Positive components show annualized market revenues, while negative components show costs and penalties.}",
            r"\label{fig:rq3-revenue-cost-decomposition}",
            r"\end{figure}",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_revenue_cost_decomposition_by_strategy(data: pd.DataFrame, out_root: Path, *, formats: set[str]) -> list[Path]:
    decomposition, component_status, unit, scale_label = extract_decomposition_rows(data)
    csv_dir = out_root / "result_section" / "csv"
    figures_dir = out_root / "result_section" / "figures"
    latex_dir = out_root / "result_section" / "latex_figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    csv_path = csv_dir / "revenue_cost_decomposition_by_market_strategy.csv"
    png_path = figures_dir / "revenue_cost_decomposition_by_market_strategy.png"
    pdf_path = figures_dir / "revenue_cost_decomposition_by_market_strategy.pdf"
    tex_path = latex_dir / "revenue_cost_decomposition_by_market_strategy.tex"

    outputs: list[Path] = []
    if "csv" in formats:
        _decomposition_long(decomposition).to_csv(csv_path, index=False)
        outputs.append(csv_path)

    unit_scale = 1.0 / 1000.0 if unit == "kEUR" else 1.0
    outputs.extend(_plot_revenue_cost_decomposition(decomposition, unit, unit_scale, png_path, pdf_path, formats=formats))
    if "tex" in formats:
        outputs.append(write_latex_revenue_cost_decomposition(decomposition, unit, unit_scale, tex_path))

    status_path = csv_dir / "revenue_cost_component_status.json"
    status_payload = {
        "component_status": component_status,
        "scale": scale_label,
    }
    if "json" in formats:
        status_path.write_text(json.dumps(status_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        outputs.append(status_path)

    for message in component_status.get("revenue", []):
        print(f"[INFO] revenue decomposition: {message}")
    for message in component_status.get("cost", []):
        print(f"[INFO] cost decomposition: {message}")

    return outputs


def generate_annualized_net_profit_by_strategy(data: pd.DataFrame, out_root: Path, *, formats: set[str]) -> list[Path]:
    csv_dir = out_root / "result_section" / "csv"
    figures_dir = out_root / "result_section" / "figures"
    latex_dir = out_root / "result_section" / "latex_figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "annualized_net_profit_by_market_strategy.csv"
    png_path = figures_dir / "annualized_net_profit_by_market_strategy.png"
    pdf_path = figures_dir / "annualized_net_profit_by_market_strategy.pdf"
    tex_path = latex_dir / "annualized_net_profit_by_market_strategy.tex"
    outputs: list[Path] = []
    if "csv" in formats:
        data.to_csv(csv_path, index=False)
        outputs.append(csv_path)
    outputs.extend(plot_annualized_net_profit(data, png_path, pdf_path, formats=formats))
    if "tex" in formats:
        outputs.append(write_latex_annualized_net_profit(data, tex_path))
    return outputs


def build_single_vs_multi_profit_uplift(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        raise ValueError("Cannot build single-vs-multi profit uplift table from empty RQ3 data.")
    by_strategy = {str(row["strategy"]): row for _, row in data.iterrows()}
    missing = [strategy for strategy in STRATEGY_ORDER if strategy not in by_strategy]
    if missing:
        raise ValueError(f"Missing strategy rows for single-vs-multi profit uplift table: {', '.join(missing)}")

    multi_profit = _safe_float(by_strategy["Multi"].get("annualized_net_profit_eur_per_year", math.nan))
    if not math.isfinite(multi_profit):
        raise ValueError("Missing or invalid annualized net profit for Multi in single-vs-multi profit uplift table.")

    rows: list[dict[str, float | str]] = []
    for strategy in STRATEGY_ORDER:
        profit = _safe_float(by_strategy[strategy].get("annualized_net_profit_eur_per_year", math.nan))
        if not math.isfinite(profit):
            raise ValueError(f"Missing or invalid annualized net profit for {strategy}.")
        if strategy == "Multi":
            absolute_uplift = math.nan
            uplift_ratio = math.nan
            uplift_percent = math.nan
        else:
            absolute_uplift = multi_profit - profit
            denominator = abs(profit)
            if denominator <= 1e-12:
                uplift_ratio = math.nan
                uplift_percent = math.nan
            else:
                uplift_ratio = absolute_uplift / denominator
                uplift_percent = 100.0 * uplift_ratio
        rows.append(
            {
                "strategy": strategy,
                "annualized_net_profit_eur_per_year": profit,
                "absolute_uplift_eur_per_year": absolute_uplift,
                "uplift_ratio": uplift_ratio,
                "uplift_percent": uplift_percent,
                "reference_strategy": "Multi",
            }
        )
    return pd.DataFrame(rows)


def _format_k_eur(value: float, *, dash_nan: bool = True) -> str:
    if not math.isfinite(float(value)):
        return "$-$" if dash_nan else "n/a"
    return f"{float(value) / 1000.0:,.1f}"


def _format_percent(value: float) -> str:
    if not math.isfinite(float(value)):
        return "$-$"
    return f"{float(value):,.1f}\\%"


def _format_k_eur_component(value: float) -> str:
    if not math.isfinite(float(value)):
        return "$-$"
    value_k = float(value) / 1000.0
    if abs(value_k) < 0.05:
        value_k = 0.0
    return f"{value_k:,.1f}"


def build_revenue_cost_component_table(data: pd.DataFrame) -> pd.DataFrame:
    decomposition, _, _, _ = extract_decomposition_rows(data)
    long = _decomposition_long(decomposition)
    rows: list[dict[str, Any]] = []
    for component_type, label in (("revenue", "Revenue"), ("cost", "Cost")):
        type_rows = long.loc[long["component_type"].eq(component_type)].copy()
        for component in type_rows["component"].drop_duplicates().tolist():
            values = type_rows.loc[type_rows["component"].eq(component)].set_index("strategy")[
                "annualized_component_eur_per_year"
            ]
            row: dict[str, Any] = {
                "component_type": label,
                "component": component,
            }
            for strategy in STRATEGY_ORDER:
                row[strategy] = _safe_float(values.get(strategy, math.nan))
            rows.append(row)
    net_row: dict[str, Any] = {"component_type": "Net", "component": "Net profit"}
    by_strategy = data.set_index("strategy")
    for strategy in STRATEGY_ORDER:
        net_row[strategy] = _safe_float(by_strategy.loc[strategy, "annualized_net_profit_eur_per_year"])
    rows.append(net_row)
    return pd.DataFrame(rows)


def write_latex_revenue_cost_component_table(table: pd.DataFrame, path: Path) -> Path:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Annualized revenue and cost components for the XGB p50 multi-market and single-market strategies. Positive values denote revenues and negative values denote costs or losses; values are shown in kEUR/year.}",
        r"\label{tab:rq3-revenue-cost-components-market-strategies}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Type} & \textbf{Component} & \textbf{Multi} & \textbf{DA-only} & \textbf{BCM-only} & \textbf{BEM-only} \\",
        r"\midrule",
    ]
    previous_type = ""
    for _, row in table.iterrows():
        component_type = str(row["component_type"])
        if previous_type and component_type != previous_type:
            if component_type == "Net":
                lines.append(r"\midrule")
            else:
                lines.append(r"\addlinespace")
        previous_type = component_type
        component = _revenue_cost_legend_label(str(row["component"]))
        if component == "Net profit":
            component_tex = r"\textbf{Net profit}"
            type_tex = r"\textbf{Net}"
        else:
            component_tex = _latex_escape(component)
            type_tex = _latex_escape(component_type)
        cells = [
            type_tex,
            component_tex,
            *[_format_k_eur_component(_safe_float(row[strategy])) for strategy in STRATEGY_ORDER],
        ]
        if component == "Net profit":
            cells = cells[:2] + [rf"\textbf{{{cell}}}" for cell in cells[2:]]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_revenue_cost_component_table(data: pd.DataFrame, out_root: Path, *, formats: set[str]) -> list[Path]:
    table = build_revenue_cost_component_table(data)
    csv_dir = out_root / "result_section" / "csv"
    latex_dir = out_root / "result_section" / "latex_tables"
    csv_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "revenue_cost_components_by_market_strategy.csv"
    tex_path = latex_dir / "revenue_cost_components_by_market_strategy.tex"
    outputs: list[Path] = []
    if "csv" in formats:
        table.to_csv(csv_path, index=False)
        outputs.append(csv_path)
    if "tex" in formats:
        outputs.append(write_latex_revenue_cost_component_table(table, tex_path))
    return outputs


def _daily_metrics_path(row: pd.Series) -> Path:
    source = Path(str(row.get("source_file", "")))
    if source.name == "performance_metrics.csv":
        return source.with_name("daily_performance_metrics.csv")
    if source.name in {"performance_metrics_all_scenarios.csv", "strategy_overview.csv", "strategy_overview_valid_only.csv"}:
        strategy_raw = str(row.get("trading_strategy", row.get("strategy", ""))).lower().replace("_", "-")
        strategy_dir = strategy_raw.split("-")[0] if strategy_raw else str(row.get("strategy", "")).lower().split("-")[0]
        candidate = source.parent / strategy_dir / "p50_p50" / "daily_performance_metrics.csv"
        if candidate.exists():
            return candidate
    return source.parent / "daily_performance_metrics.csv"


def _backtest_summary_path(row: pd.Series) -> Path:
    source = Path(str(row.get("source_file", "")))
    if source.name == "performance_metrics.csv":
        return source.with_name("backtest_summary.json")
    strategy_raw = str(row.get("trading_strategy", row.get("strategy", ""))).lower().replace("_", "-")
    strategy_dir = strategy_raw.split("-")[0] if strategy_raw else str(row.get("strategy", "")).lower().split("-")[0]
    candidate = source.parent / strategy_dir / "p50_p50" / "backtest_summary.json"
    if candidate.exists():
        return candidate
    return source.parent / "backtest_summary.json"


def _load_backtest_summary(row: pd.Series) -> dict[str, Any]:
    path = _backtest_summary_path(row)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _daily_risk_stats(daily: pd.DataFrame) -> dict[str, float]:
    if "net_revenue_eur" not in daily.columns:
        raise ValueError("daily_performance_metrics.csv is missing net_revenue_eur.")
    pnl = pd.to_numeric(daily["net_revenue_eur"], errors="coerce").dropna()
    if pnl.empty:
        raise ValueError("daily_performance_metrics.csv has no finite net_revenue_eur values.")
    cumulative = pnl.cumsum()
    drawdown = cumulative - cumulative.cummax()
    q05 = pnl.quantile(0.05)
    tail = pnl.loc[pnl <= q05]
    fallback_hours = (
        float(pd.to_numeric(daily["fallback_hours"], errors="coerce").fillna(0.0).sum())
        if "fallback_hours" in daily.columns
        else math.nan
    )
    n_hours = (
        float(pd.to_numeric(daily["n_hours"], errors="coerce").fillna(0.0).sum())
        if "n_hours" in daily.columns
        else math.nan
    )
    return {
        "n_days": float(len(pnl)),
        "loss_day_share_percent": float((pnl < 0.0).mean() * 100.0),
        "worst_day_eur": float(pnl.min()),
        "daily_pnl_cvar_5_eur": float(tail.mean()) if not tail.empty else math.nan,
        "max_drawdown_eur": float(drawdown.min()),
        "profit_volatility_eur": float(pnl.std(ddof=0)),
        "daily_fallback_hours": fallback_hours,
        "daily_hours": n_hours,
        "daily_fallback_share_percent": float(100.0 * fallback_hours / n_hours) if n_hours and n_hours > 0 else math.nan,
    }


def build_risk_robustness_market_strategy_table(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        strategy = str(row["strategy"])
        daily_path = _daily_metrics_path(row)
        if not daily_path.exists():
            raise FileNotFoundError(f"Missing daily performance metrics for {strategy}: {daily_path}")
        daily = pd.read_csv(daily_path)
        risk = _daily_risk_stats(daily)
        summary = _load_backtest_summary(row)
        fallback_share = _safe_float(summary.get("optimizer_fallback_share_pct", risk["daily_fallback_share_percent"]))
        fallback_hours = _safe_float(summary.get("optimizer_fallback_hours", risk["daily_fallback_hours"]))
        annualization_factor = _safe_float(row.get("annualization_factor", math.nan))
        if not math.isfinite(annualization_factor):
            days = _safe_float(row.get("n_days", row.get("duration_days", math.nan)))
            annualization_factor = 365.0 / days if days > 0 else math.nan
        penalty = _safe_float(row.get("penalty_cost_eur", summary.get("total_penalty_cost_eur", math.nan)))
        terminal_repair = _safe_float(
            row.get("terminal_soc_repair_cost_eur", summary.get("terminal_soc_repair_cost_eur", math.nan))
        )
        rows.append(
            {
                "strategy": strategy,
                "annualized_net_profit_eur": _safe_float(row.get("annualized_net_profit_eur_per_year")),
                "loss_day_share_percent": risk["loss_day_share_percent"],
                "worst_day_eur": risk["worst_day_eur"],
                "max_drawdown_eur": risk["max_drawdown_eur"],
                "daily_pnl_cvar_5_eur": risk["daily_pnl_cvar_5_eur"],
                "profit_volatility_eur": risk["profit_volatility_eur"],
                "fallback_hours": fallback_hours,
                "fallback_share_percent": fallback_share,
                "annualized_penalty_cost_eur": penalty * annualization_factor if math.isfinite(penalty) and math.isfinite(annualization_factor) else math.nan,
                "annualized_terminal_soc_repair_cost_eur": terminal_repair * annualization_factor if math.isfinite(terminal_repair) and math.isfinite(annualization_factor) else math.nan,
                "thesis_reportable": _safe_float(row.get("thesis_reportable", math.nan)),
                "invalid_reason": "" if pd.isna(row.get("invalid_reason", "")) else str(row.get("invalid_reason", "")),
                "daily_metrics_path": str(daily_path),
                "backtest_summary_path": str(_backtest_summary_path(row)),
            }
        )
    out = pd.DataFrame(rows)
    out["_strategy_order"] = out["strategy"].map({name: idx for idx, name in enumerate(STRATEGY_ORDER)})
    return out.sort_values("_strategy_order").drop(columns="_strategy_order")


def _format_daily_k_eur(value: Any) -> str:
    x = _safe_float(value)
    if not math.isfinite(x):
        return "$-$"
    return f"{x / 1000.0:,.2f}"


def _format_table_percent_from_percent(value: Any, digits: int = 1) -> str:
    x = _safe_float(value)
    if not math.isfinite(x):
        return "$-$"
    return f"{x:,.{digits}f}\\%"


def _format_yes_no(value: Any) -> str:
    x = _safe_float(value)
    if not math.isfinite(x):
        return "$-$"
    return "Yes" if x >= 0.5 else "No"


def write_latex_risk_robustness_market_strategy_table(table: pd.DataFrame, path: Path) -> Path:
    strategy_labels = {
        "Multi": "Multi-market",
        "DA-only": "DA-only",
        "BCM-only": "BCM-only",
        "BEM-only": "BEM-only",
    }
    by_strategy = {str(row["strategy"]): row for _, row in table.iterrows()}
    strategies = [strategy for strategy in STRATEGY_ORDER if strategy in by_strategy]

    metric_rows: list[tuple[str, Callable[[pd.Series], str]]] = [
        ("Net profit (kEUR/y)", lambda row: _format_k_eur_component(_safe_float(row["annualized_net_profit_eur"]))),
        ("Loss-day share", lambda row: _format_table_percent_from_percent(row["loss_day_share_percent"], 1)),
        ("Worst day (kEUR)", lambda row: _format_daily_k_eur(row["worst_day_eur"])),
        ("Max drawdown (kEUR)", lambda row: _format_daily_k_eur(row["max_drawdown_eur"])),
        ("CVaR 5% (kEUR)", lambda row: _format_daily_k_eur(row["daily_pnl_cvar_5_eur"])),
        ("Volatility (kEUR)", lambda row: _format_daily_k_eur(row["profit_volatility_eur"])),
        ("Fallback share", lambda row: _format_table_percent_from_percent(row["fallback_share_percent"], 1)),
        ("Penalty (kEUR/y)", lambda row: _format_k_eur_component(_safe_float(row["annualized_penalty_cost_eur"]))),
        (
            "Terminal repair (kEUR/y)",
            lambda row: _format_k_eur_component(_safe_float(row["annualized_terminal_soc_repair_cost_eur"])),
        ),
        ("Reportable", lambda row: _format_yes_no(row["thesis_reportable"])),
    ]

    column_spec = "l" + ("r" * len(strategies))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Risk and robustness diagnostics for the XGB p50 multi-market and single-market strategies. Daily risk values are computed from test-period daily net revenue; profit, penalty and terminal repair values are annualized and reported in kEUR/year.}",
        r"\label{tab:rq3-risk-robustness-market-strategies}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join([r"\textbf{Metric}", *[rf"\textbf{{{_latex_escape(strategy_labels.get(strategy, strategy))}}}" for strategy in strategies]]) + r" \\",
        r"\midrule",
    ]
    for label, formatter in metric_rows:
        cells = [_latex_escape(label), *[formatter(by_strategy[strategy]) for strategy in strategies]]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            (
                r"\par\vspace{0.3em}\footnotesize\raggedright "
                r"CVaR 5\% is the mean of days in the lower 5\% tail of daily net revenue. "
                r"Fallback share is based on optimizer fallback hours. "
            ),
            r"\end{table}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_risk_robustness_market_strategy_table(data: pd.DataFrame, out_root: Path, *, formats: set[str]) -> list[Path]:
    table = build_risk_robustness_market_strategy_table(data)
    csv_dir = out_root / "result_section" / "csv"
    latex_dir = out_root / "result_section" / "latex_tables"
    csv_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "risk_robustness_by_market_strategy.csv"
    tex_path = latex_dir / "risk_robustness_by_market_strategy.tex"
    outputs: list[Path] = []
    if "csv" in formats:
        table.to_csv(csv_path, index=False)
        outputs.append(csv_path)
    if "tex" in formats:
        outputs.append(write_latex_risk_robustness_market_strategy_table(table, tex_path))
    return outputs


def write_latex_single_vs_multi_profit_uplift(table: pd.DataFrame, path: Path) -> Path:
    rows = []
    for _, row in table.iterrows():
        rows.append(
            " & ".join(
                [
                    _latex_escape(str(row["strategy"])),
                    _format_k_eur(_safe_float(row["annualized_net_profit_eur_per_year"])),
                    _format_k_eur(_safe_float(row["absolute_uplift_eur_per_year"])),
                    _format_percent(_safe_float(row["uplift_percent"])),
                ]
            )
            + r" \\"
        )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Annualized net profit and profit uplift of the XGB p50 multi-market strategy relative to single-market strategies. Uplift is computed as $(\Pi_{\mathrm{multi}}-\Pi_{\mathrm{single}})/|\Pi_{\mathrm{single}}|$.}",
        r"\label{tab:rq3-single-vs-multi-market-profit-uplift}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Strategy} & \textbf{\shortstack{Net profit\\(kEUR/year)}} & \textbf{\shortstack{Absolute uplift\\(kEUR/year)}} & \textbf{Uplift (\%)} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_single_vs_multi_profit_uplift_table(data: pd.DataFrame, out_root: Path, *, formats: set[str]) -> list[Path]:
    table = build_single_vs_multi_profit_uplift(data)
    csv_dir = out_root / "result_section" / "csv"
    latex_dir = out_root / "result_section" / "latex_tables"
    csv_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "single_vs_multi_market_profit_uplift.csv"
    tex_path = latex_dir / "single_vs_multi_market_profit_uplift.tex"
    outputs: list[Path] = []
    if "csv" in formats:
        table.to_csv(csv_path, index=False)
        outputs.append(csv_path)
    if "tex" in formats:
        outputs.append(write_latex_single_vs_multi_profit_uplift(table, tex_path))
    return outputs


def build_cumulative_net_profit_by_strategy(run_root: Path) -> tuple[pd.DataFrame, Path, str]:
    loaded = _load_cumulative_series_candidate(run_root)
    if loaded is None:
        raise FileNotFoundError(
            "No suitable timestamp-level RQ3 net-profit source was found for Figure 3. "
            f"Looked under {run_root} for daily/hourly/path PnL files."
        )
    raw, selected_path = loaded
    raw = raw.copy()
    preferred_source_by_strategy = {
        "Multi": "/xgb_multi_p50/",
        "DA-only": "/xgb_da_p50/",
        "BCM-only": "/xgb_bcm_p50/",
        "BEM-only": "/xgb_bem_p50/",
    }
    preferred_mask = raw.apply(
        lambda row: preferred_source_by_strategy.get(str(row.get("strategy")), "") in str(row.get("source_file", "")),
        axis=1,
    )
    if preferred_mask.any():
        raw = raw.loc[preferred_mask].copy()
    raw["date"] = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="coerce").dt.floor("D")
    raw["raw_pnl_eur"] = pd.to_numeric(raw["raw_pnl_eur"], errors="coerce")
    raw = raw.dropna(subset=["date", "strategy", "raw_pnl_eur"]).copy()
    if raw.empty:
        raise ValueError(f"Selected cumulative source has no usable timestamp/PnL rows: {selected_path}")

    if raw["is_cumulative_input"].astype(bool).all():
        daily = (
            raw.sort_values("timestamp_utc")
            .groupby(["strategy", "date"], as_index=False)
            .agg(
                cumulative_net_profit_eur=("raw_pnl_eur", "last"),
                source_file=("source_file", "first"),
            )
        )
        daily["daily_net_profit_eur"] = daily.groupby("strategy")["cumulative_net_profit_eur"].diff()
        first_mask = daily.groupby("strategy").cumcount().eq(0)
        daily.loc[first_mask, "daily_net_profit_eur"] = daily.loc[first_mask, "cumulative_net_profit_eur"]
        resolution = "daily from cumulative source"
    else:
        daily = (
            raw.groupby(["strategy", "date"], as_index=False)
            .agg(
                daily_net_profit_eur=("raw_pnl_eur", "sum"),
                source_file=("source_file", "first"),
            )
            .sort_values(["strategy", "date"])
        )
        daily["cumulative_net_profit_eur"] = daily.groupby("strategy")["daily_net_profit_eur"].cumsum()
        resolution = "daily aggregated from period net profit"

    daily = daily.loc[daily["strategy"].isin(STRATEGY_ORDER)].copy()
    missing = [strategy for strategy in STRATEGY_ORDER if strategy not in set(daily["strategy"])]
    if missing:
        detected = sorted(set(daily["strategy"]))
        raise ValueError(
            "Cumulative RQ3 series does not contain all required strategies. "
            f"Missing={missing}; detected={detected}; selected_source={selected_path}"
        )
    daily["_strategy_order"] = daily["strategy"].map({name: idx for idx, name in enumerate(STRATEGY_ORDER)})
    daily = daily.sort_values(["_strategy_order", "date"]).drop(columns=["_strategy_order"])
    daily["date"] = pd.to_datetime(daily["date"], utc=True).dt.date.astype(str)
    return daily, selected_path, resolution


def plot_cumulative_net_profit_by_strategy(data: pd.DataFrame, unit: str, unit_scale: float, png_path: Path, pdf_path: Path, *, formats: set[str]) -> list[Path]:
    apply_geo_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for strategy in STRATEGY_ORDER:
        d = data.loc[data["strategy"].eq(strategy)].copy()
        if d.empty:
            continue
        dates = pd.to_datetime(d["date"], errors="coerce")
        values = pd.to_numeric(d["cumulative_net_profit_eur"], errors="coerce") * unit_scale
        ax.plot(
            dates,
            values,
            label=strategy,
            color=STRATEGY_COLOR.get(strategy, THESIS_PALETTE["neutral_dark"]),
            linewidth=2.0,
        )
    ax.axhline(0.0, color=THESIS_PALETTE["neutral_dark"], linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Cumulative net profit ({unit})")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper left", frameon=False, ncol=2)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if "png" in formats:
        fig.savefig(png_path, dpi=220)
        paths.append(png_path)
    if "pdf" in formats:
        fig.savefig(pdf_path)
        paths.append(pdf_path)
    plt.close(fig)
    return paths


def write_latex_cumulative_net_profit_by_strategy(data: pd.DataFrame, unit: str, unit_scale: float, path: Path) -> Path:
    dates = sorted(pd.to_datetime(data["date"], errors="coerce").dropna().unique())
    if not dates:
        raise ValueError("Cannot write cumulative RQ3 LaTeX figure without dates.")
    date_index = {pd.Timestamp(date).date().isoformat(): idx for idx, date in enumerate(dates)}
    x_tick_step = max(1, math.ceil(len(dates) / 6))
    x_ticks = list(range(0, len(dates), x_tick_step))
    if x_ticks[-1] != len(dates) - 1:
        x_ticks.append(len(dates) - 1)
    x_tick_labels = [pd.Timestamp(dates[idx]).strftime("%d %b") for idx in x_ticks]
    values = pd.to_numeric(data["cumulative_net_profit_eur"], errors="coerce") * unit_scale
    ymin = min(0.0, float(values.min())) if values.notna().any() else -1.0
    ymax = max(0.0, float(values.max())) if values.notna().any() else 1.0
    span = max(1.0, ymax - ymin)

    lines = [
        r"% Requires \usepackage{pgfplots}",
        r"% Requires \pgfplotsset{compat=1.18}",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        _tex_color_def("rqThreeNeutral", THESIS_PALETTE["neutral_dark"]),
        *[
            _tex_color_def(f"rqThreeLine{strategy.replace('-', '').replace('only', 'Only')}", STRATEGY_COLOR.get(strategy, THESIS_PALETTE["neutral_dark"]))
            for strategy in STRATEGY_ORDER
        ],
        r"\begin{axis}[",
        r"tick align=outside,",
        r"axis line style={rqThreeNeutral},",
        r"tick style={rqThreeNeutral},",
        r"label style={font=\small},",
        r"tick label style={font=\small},",
        r"grid=major,",
        r"grid style={black!12, line width=0.2pt},",
        r"width=0.88\linewidth,",
        r"height=0.48\linewidth,",
        r"xlabel={Date},",
        rf"ylabel={{Cumulative net profit ({_latex_escape(unit)})}},",
        f"xmin=0, xmax={len(dates) - 1},",
        f"xtick={{{','.join(str(i) for i in x_ticks)}}},",
        f"xticklabels={{{','.join(_latex_escape(label) for label in x_tick_labels)}}},",
        rf"ymin={_tex_float(ymin - 0.08 * span, 4)}, ymax={_tex_float(ymax + 0.10 * span, 4)},",
        r"legend columns=2,",
        r"legend cell align=left,",
        r"legend style={at={(0.02,0.98)}, anchor=north west, font=\small, draw=none, fill=none, /tikz/every even column/.append style={column sep=0.35cm}},",
        r"]",
    ]
    for strategy in STRATEGY_ORDER:
        d = data.loc[data["strategy"].eq(strategy)].copy()
        coords: list[str] = []
        for _, row in d.iterrows():
            date_key = str(row["date"])
            if date_key not in date_index:
                continue
            y = _safe_float(row["cumulative_net_profit_eur"]) * unit_scale
            if math.isfinite(y):
                coords.append(f"({date_index[date_key]},{_tex_float(y, 4)})")
        color_name = f"rqThreeLine{strategy.replace('-', '').replace('only', 'Only')}"
        lines.extend(
            [
                rf"\addplot+[mark=none, line width=1.2pt, color={color_name}] coordinates {{{' '.join(coords)}}};",
                rf"\addlegendentry{{{_latex_escape(strategy)}}}",
            ]
        )
    lines.extend(
        [
            rf"\draw[rqThreeNeutral, line width=0.6pt] (axis cs:0,0) -- (axis cs:{len(dates) - 1},0);",
            r"\end{axis}",
            r"\end{tikzpicture}",
            r"\caption{Cumulative net profit by market participation strategy over the test period. The figure compares the XGB p50 multi-market strategy with DA-only, BCM-only and BEM-only baselines and shows whether profitability differences develop persistently or are driven by individual high-revenue periods.}",
            r"\label{fig:rq3-cumulative-net-profit-by-market-strategy}",
            r"\end{figure}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_cumulative_net_profit_by_strategy(run_root: Path, out_root: Path, *, formats: set[str]) -> tuple[list[Path], pd.DataFrame, Path, str]:
    cumulative, source_path, resolution = build_cumulative_net_profit_by_strategy(run_root)
    csv_dir = out_root / "result_section" / "csv"
    figures_dir = out_root / "result_section" / "figures"
    latex_dir = out_root / "result_section" / "latex_figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "cumulative_net_profit_by_market_strategy.csv"
    png_path = figures_dir / "cumulative_net_profit_by_market_strategy.png"
    pdf_path = figures_dir / "cumulative_net_profit_by_market_strategy.pdf"
    tex_path = latex_dir / "cumulative_net_profit_by_market_strategy.tex"
    outputs: list[Path] = []
    if "csv" in formats:
        cumulative.to_csv(csv_path, index=False)
        outputs.append(csv_path)
    unit, unit_scale = _choose_scale(cumulative["cumulative_net_profit_eur"].tolist())
    outputs.extend(plot_cumulative_net_profit_by_strategy(cumulative, unit, unit_scale, png_path, pdf_path, formats=formats))
    if "tex" in formats:
        outputs.append(write_latex_cumulative_net_profit_by_strategy(cumulative, unit, unit_scale, tex_path))
    return outputs, cumulative, source_path, resolution


def _operational_metric_value(row: pd.Series, spec: dict[str, Any]) -> tuple[float, str, str]:
    columns = list(row.index)
    for candidate in spec["candidates"]:
        method_suffix = "direct"
        if "+" in candidate:
            parts = tuple(p.strip() for p in str(candidate).split("+"))
            if all(part in row.index for part in parts):
                values = [_safe_float(row[part]) for part in parts]
                if all(math.isfinite(v) for v in values):
                    value = sum(abs(v) for v in values)
                    source_col = "+".join(parts)
                    method_suffix = "sum_abs_components"
                    break
            continue
        else:
            col = _resolve_exact_matching_column(columns, (str(candidate),))
            if col is None:
                continue
            value = _safe_float(row[col])
            source_col = col
            if not math.isfinite(value):
                continue
            if str(candidate).startswith("annualized_"):
                method_suffix = "annualized_value"
            break
    else:
        return math.nan, ",".join(str(c) for c in spec["candidates"][:3]), "missing"

    if bool(spec.get("annualize")) and method_suffix != "annualized_value":
        annualization_factor = _safe_float(row.get("annualization_factor", math.nan))
        if not math.isfinite(annualization_factor) or annualization_factor <= 0:
            return math.nan, f"{source_col},annualization_factor", "missing_annualization_factor"
        value *= annualization_factor
        method_suffix = f"annualized_{method_suffix}"
    scale = _safe_float(spec.get("scale", 1.0))
    if math.isfinite(scale):
        value *= scale
    return value, source_col, method_suffix


def _format_operational_annotation(value: float, unit: str) -> str:
    if not math.isfinite(value):
        return "--"
    if unit == "%":
        pct = value * 100.0 if abs(value) <= 1.0 else value
        return f"{pct:.1f}%"
    if unit in {"EUR", "EUR/year"}:
        if abs(value) >= 1000.0:
            return f"{value / 1000.0:,.0f}k"
        return f"{value:,.0f}"
    if unit in {"MWh", "MWh/day", "MWh/year", "GWh/year", "kEUR/year", "cycles", "cycles/day"}:
        if abs(value) >= 100.0:
            return f"{value:,.0f}"
        if abs(value) >= 10.0:
            return f"{value:,.1f}"
        return f"{value:,.2f}"
    return f"{value:,.1f}"


def _operational_display_metric(metric: str) -> str:
    for spec in OPERATIONAL_METRIC_SPECS:
        if spec["metric"] == metric:
            return str(spec.get("display_metric", metric))
    return metric


def build_operational_intensity_by_strategy(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    omitted: list[str] = []
    for spec in OPERATIONAL_METRIC_SPECS:
        metric_rows: list[dict[str, Any]] = []
        missing_by_strategy: list[str] = []
        for _, row in data.iterrows():
            strategy = str(row["strategy"])
            value, source_col, method = _operational_metric_value(row, spec)
            if math.isfinite(value):
                metric_rows.append(
                    {
                        "strategy": strategy,
                        "metric": spec["metric"],
                        "value": value,
                        "unit": spec["unit"],
                        "source_column": source_col,
                        "method": method,
                    }
                )
            else:
                missing_by_strategy.append(strategy)
        if metric_rows:
            rows.extend(metric_rows)
            if missing_by_strategy:
                omitted.append(
                    f"{spec['metric']}: missing for {', '.join(missing_by_strategy)} "
                    f"(candidates: {', '.join(spec['candidates'])})"
                )
        else:
            omitted.append(f"{spec['metric']}: omitted; no candidate columns found ({', '.join(spec['candidates'])})")

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(
            "No operational-intensity metrics were found in the selected RQ3 summary rows. "
            "Expected columns include throughput_mwh_per_day, equivalent_full_cycles_total, "
            "mean_soc_mwh, realized_aux_cost_eur, realized_degradation_cost_eur or id_abs_mwh_total."
        )
    out["_strategy_order"] = out["strategy"].map({name: idx for idx, name in enumerate(STRATEGY_ORDER)})
    out["_metric_order"] = out["metric"].map({spec["metric"]: idx for idx, spec in enumerate(OPERATIONAL_METRIC_SPECS)})
    out = out.sort_values(["_strategy_order", "_metric_order"]).drop(columns=["_strategy_order", "_metric_order"])

    norm_values = pd.Series(np.nan, index=out.index, dtype=float)
    for metric, group in out.groupby("metric", sort=False):
        values = pd.to_numeric(group["value"], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            norm = pd.Series(np.nan, index=group.index)
        else:
            min_v = float(finite.min())
            max_v = float(finite.max())
            if math.isclose(max_v, min_v):
                fill = 1.0 if max_v > 0 else 0.0
                norm = pd.Series(fill, index=group.index)
            else:
                norm = (values - min_v) / (max_v - min_v)
        norm_values.loc[group.index] = norm
    out["normalized_value"] = norm_values
    out["display_value"] = [_format_operational_annotation(v, u) for v, u in zip(out["value"], out["unit"])]
    return out, omitted


def _operational_matrix(data: pd.DataFrame) -> tuple[list[str], list[str], np.ndarray, list[list[str]]]:
    metrics = [spec["metric"] for spec in OPERATIONAL_METRIC_SPECS if spec["metric"] in set(data["metric"])]
    matrix = np.full((len(STRATEGY_ORDER), len(metrics)), np.nan, dtype=float)
    annotations = [["" for _ in metrics] for _ in STRATEGY_ORDER]
    for i, strategy in enumerate(STRATEGY_ORDER):
        for j, metric in enumerate(metrics):
            d = data.loc[data["strategy"].eq(strategy) & data["metric"].eq(metric)]
            if d.empty:
                continue
            row = d.iloc[0]
            matrix[i, j] = _safe_float(row["normalized_value"])
            annotations[i][j] = str(row["display_value"])
    return STRATEGY_ORDER, metrics, matrix, annotations


def plot_operational_intensity_by_strategy(data: pd.DataFrame, png_path: Path, pdf_path: Path, *, formats: set[str]) -> list[Path]:
    apply_geo_style()
    strategies, metrics, matrix, annotations = _operational_matrix(data)
    metric_labels = [_operational_display_metric(metric) for metric in metrics]
    cmap = LinearSegmentedColormap.from_list(
        "rq3_operational_intensity",
        ["#FFFFFF", "#E4F1F7", THESIS_PALETTE["primary"]],
    )
    fig_width = max(9.8, 1.15 * len(metrics) + 3.2)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(metrics)), labels=metric_labels, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(strategies)), labels=strategies)
    ax.tick_params(axis="both", length=0, labelsize=10)
    ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(strategies), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(len(strategies)):
        for j in range(len(metrics)):
            if not math.isfinite(matrix[i, j]):
                continue
            color = "white" if matrix[i, j] >= 0.62 else THESIS_PALETTE["neutral_dark"]
            ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=10, color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.038, pad=0.045)
    cbar.set_label("Relative operational intensity", fontsize=10, labelpad=12)
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if "png" in formats:
        fig.savefig(png_path, dpi=220)
        paths.append(png_path)
    if "pdf" in formats:
        fig.savefig(pdf_path)
        paths.append(pdf_path)
    plt.close(fig)
    return paths


def write_latex_operational_intensity_by_strategy(data: pd.DataFrame, path: Path) -> Path:
    strategies, metrics, matrix, annotations = _operational_matrix(data)
    metric_tick_labels = {
        "Throughput": r"{\shortstack{Throughput\\in GWh}}",
        "Mean SoC": r"{\shortstack{Mean SoC\\in MWh}}",
        "Auxiliary cost": r"{\shortstack{Auxiliary\\cost\\in kEUR}}",
        "Degradation cost": r"{\shortstack{Degra-\\dation cost\\in kEUR}}",
        "ID recourse volume": r"{\shortstack{ID recourse\\volume\\in MWh}}",
        "SoC violation count": r"{\shortstack{SoC violation\\count}}",
    }
    cell_lines: list[str] = []
    for i, _strategy in enumerate(strategies):
        for j, _metric in enumerate(metrics):
            value = float(matrix[i, j]) if math.isfinite(matrix[i, j]) else math.nan
            if not math.isfinite(value):
                continue
            fill_pct = int(round(max(0.0, min(1.0, value)) * 100.0))
            text_color = "white" if value >= 0.62 else "rqThreeNeutral"
            cell_lines.append(
                rf"\filldraw[fill=rqThreeIntensity!{fill_pct}!white, draw=white, line width=0.4pt] "
                rf"(axis cs:{_tex_float(j - 0.5, 3)},{_tex_float(i - 0.5, 3)}) rectangle "
                rf"(axis cs:{_tex_float(j + 0.5, 3)},{_tex_float(i + 0.5, 3)});"
            )
            cell_lines.append(
                rf"\node[font=\small, text={text_color}] at (axis cs:{j},{i}) "
                rf"{{{_latex_escape(annotations[i][j])}}};"
            )

    scale_left = len(metrics) - 0.02
    scale_right = len(metrics) + 0.14
    scale_tick_x = len(metrics) + 0.22
    scale_label_x = len(metrics) + 0.88
    axis_right = len(metrics) + 1.10
    scale_top = -0.5
    scale_mid = (len(strategies) - 1) / 2.0
    scale_bottom = len(strategies) - 0.5
    scale_label_top = scale_top + 0.18
    scale_label_bottom = scale_bottom - 0.18

    lines = [
        r"% Requires \usepackage{pgfplots}",
        r"% Requires \pgfplotsset{compat=1.18}",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        _tex_color_def("rqThreeNeutral", THESIS_PALETTE["neutral_dark"]),
        _tex_color_def("rqThreeIntensity", THESIS_PALETTE["primary"]),
        r"\begin{axis}[",
        r"tick align=outside,",
        r"axis line style={draw=none},",
        r"tick style={draw=none},",
        r"label style={font=\small},",
        r"tick label style={font=\small},",
        r"width=1.00\linewidth,",
        r"height=0.46\linewidth,",
        rf"xmin=-0.5, xmax={_tex_float(axis_right, 3)},",
        rf"ymin=-0.5, ymax={_tex_float(len(strategies) - 0.5, 3)},",
        r"y dir=reverse,",
        "xtick={" + ",".join(str(i) for i in range(len(metrics))) + "},",
        "xticklabels={" + ",".join(metric_tick_labels.get(m, _latex_escape(m)) for m in metrics) + "},",
        r"xticklabel style={font=\small, anchor=north, align=center, yshift=-0.2em},",
        "ytick={" + ",".join(str(i) for i in range(len(strategies))) + "},",
        "yticklabels={" + ",".join(_latex_escape(s) for s in strategies) + "},",
        r"yticklabel style={font=\small, align=right},",
        r"]",
        *cell_lines,
        rf"\shade[bottom color=white,top color=rqThreeIntensity] (axis cs:{_tex_float(scale_left, 3)},{_tex_float(scale_bottom, 3)}) rectangle (axis cs:{_tex_float(scale_right, 3)},{_tex_float(scale_top, 3)});",
        rf"\draw[black!35, line width=0.25pt] (axis cs:{_tex_float(scale_left, 3)},{_tex_float(scale_bottom, 3)}) rectangle (axis cs:{_tex_float(scale_right, 3)},{_tex_float(scale_top, 3)});",
        rf"\draw[black!35, line width=0.25pt] (axis cs:{_tex_float(scale_right, 3)},{_tex_float(scale_bottom, 3)}) -- (axis cs:{_tex_float(scale_right + 0.06, 3)},{_tex_float(scale_bottom, 3)});",
        rf"\draw[black!35, line width=0.25pt] (axis cs:{_tex_float(scale_right, 3)},{_tex_float(scale_mid, 3)}) -- (axis cs:{_tex_float(scale_right + 0.07, 3)},{_tex_float(scale_mid, 3)});",
        rf"\draw[black!35, line width=0.25pt] (axis cs:{_tex_float(scale_right, 3)},{_tex_float(scale_top, 3)}) -- (axis cs:{_tex_float(scale_right + 0.06, 3)},{_tex_float(scale_top, 3)});",
        rf"\node[anchor=west, font=\small] at (axis cs:{_tex_float(scale_tick_x, 3)},{_tex_float(scale_label_bottom, 3)}) {{0}};",
        rf"\node[anchor=west, font=\small] at (axis cs:{_tex_float(scale_tick_x, 3)},{_tex_float(scale_mid, 3)}) {{0.5}};",
        rf"\node[anchor=west, font=\small] at (axis cs:{_tex_float(scale_tick_x, 3)},{_tex_float(scale_label_top, 3)}) {{1}};",
        rf"\node[anchor=south, rotate=90, font=\small] at (axis cs:{_tex_float(scale_label_x, 3)},{_tex_float(scale_mid, 3)}) {{Relative operational intensity}};",
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\caption{Operational intensity by market strategy, comparing normalized battery use, costs and recourse metrics for XGB p50 multi-market and single-market strategies. Throughput, auxiliary cost, degradation cost and ID recourse volume are annualized.}",
        r"\label{fig:rq3-operational-intensity-by-market-strategy}",
        r"\end{figure}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_operational_intensity_by_strategy(data: pd.DataFrame, out_root: Path, *, formats: set[str]) -> tuple[list[Path], pd.DataFrame, list[str]]:
    operational, omitted = build_operational_intensity_by_strategy(data)
    csv_dir = out_root / "result_section" / "csv"
    figures_dir = out_root / "result_section" / "figures"
    latex_dir = out_root / "result_section" / "latex_figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "operational_intensity_by_market_strategy.csv"
    png_path = figures_dir / "operational_intensity_by_market_strategy.png"
    pdf_path = figures_dir / "operational_intensity_by_market_strategy.pdf"
    tex_path = latex_dir / "operational_intensity_by_market_strategy.tex"
    outputs: list[Path] = []
    if "csv" in formats:
        operational.to_csv(csv_path, index=False)
        outputs.append(csv_path)
    outputs.extend(plot_operational_intensity_by_strategy(operational, png_path, pdf_path, formats=formats))
    if "tex" in formats:
        outputs.append(write_latex_operational_intensity_by_strategy(operational, tex_path))
    return outputs, operational, omitted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=None, help="Explicit RQ3 simulation output root.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="RQ3 benchmark output root.")
    parser.add_argument(
        "--formats",
        default="csv,json,png,pdf,tex",
        help="Comma-separated output formats to write. Use 'png,tex' for fast thesis-figure regeneration.",
    )
    parser.add_argument("--skip-pdf", action="store_true", help="Do not write PDF figures.")
    parser.add_argument("--skip-csv", action="store_true", help="Do not write CSV outputs.")
    parser.add_argument("--skip-json", action="store_true", help="Do not write JSON diagnostics/status outputs.")
    parser.add_argument("--skip-png", action="store_true", help="Do not write PNG figures.")
    parser.add_argument("--skip-tex", action="store_true", help="Do not write LaTeX figure files.")
    parser.add_argument("--export-dir", type=Path, default=Path(DEFAULT_EXPORT_DIR), help="Thesis export destination.")
    parser.add_argument("--skip-export", action="store_true", help="Do not export the generated RQ3 folder to the thesis figures directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    formats = _parse_formats(args.formats)
    if args.skip_pdf:
        formats.discard("pdf")
    if args.skip_csv:
        formats.discard("csv")
    if args.skip_json:
        formats.discard("json")
    if args.skip_png:
        formats.discard("png")
    if args.skip_tex:
        formats.discard("tex")
    if not formats:
        raise ValueError("No output formats remain after applying skip flags.")
    candidate = discover_input(args.run_root)
    data = select_strategy_rows(candidate)
    decomp_data, component_status, unit, scale_label = extract_decomposition_rows(data)
    outputs = []
    outputs.extend(generate_annualized_net_profit_by_strategy(data, args.out_root, formats=formats))
    outputs.extend(generate_single_vs_multi_profit_uplift_table(data, args.out_root, formats=formats))
    outputs.extend(generate_revenue_cost_decomposition_by_strategy(data, args.out_root, formats=formats))
    outputs.extend(generate_revenue_cost_component_table(data, args.out_root, formats=formats))
    outputs.extend(generate_risk_robustness_market_strategy_table(data, args.out_root, formats=formats))
    outputs.extend(generate_cleared_bid_volume_market_comparison(data, args.out_root, formats=formats))
    outputs.extend(generate_bcm_revenue_mechanism_comparison(data, args.out_root, formats=formats))
    cumulative_outputs, cumulative_data, cumulative_source, cumulative_resolution = generate_cumulative_net_profit_by_strategy(candidate.root, args.out_root, formats=formats)
    outputs.extend(cumulative_outputs)
    operational_outputs, operational_data, operational_omitted = generate_operational_intensity_by_strategy(data, args.out_root, formats=formats)
    outputs.extend(operational_outputs)

    print(f"[OK] selected input root: {candidate.root}")
    print(f"[OK] selected cumulative input file: {cumulative_source}")
    print("[OK] selected source files:")
    for path in candidate.source_files:
        print(f"  - {path}")
    detected_strategies = [s for s in STRATEGY_ORDER if s in set(data["strategy"])]
    revenue_components = _revenue_components_for_plot(decomp_data)
    cost_components = _cost_components_for_plot(decomp_data)
    print(f"[OK] detected strategies: {', '.join(detected_strategies)}")
    print(
        f"[OK] detected revenue components: {', '.join(revenue_components) if revenue_components else 'none'}"
    )
    print(f"[OK] detected cost components: {', '.join(cost_components) if cost_components else 'none'}")
    detected_operational_metrics = [m for m in [spec["metric"] for spec in OPERATIONAL_METRIC_SPECS] if m in set(operational_data["metric"])]
    print(
        f"[OK] detected operational metrics: "
        f"{', '.join(detected_operational_metrics) if detected_operational_metrics else 'none'}"
    )
    if operational_omitted:
        print("[OK] omitted operational metrics:")
        for message in operational_omitted:
            print(f"  - {message}")
    print(f"[OK] annualization method: {decomp_data['annualization_method'].iloc[0] if not decomp_data.empty else 'n/a'}")
    print(f"[OK] output decomposition unit: {scale_label}")
    print(f"[OK] cumulative time resolution: {cumulative_resolution}")
    if not cumulative_data.empty:
        print(f"[OK] cumulative date range: {cumulative_data['date'].min()} -> {cumulative_data['date'].max()}")
    print("[OK] annualized net profit by strategy:")
    for _, row in data.iterrows():
        print(f"  - {row['strategy']}: {row['annualized_net_profit_eur_per_year']:,.2f} EUR/year")
    print("[OK] final cumulative net profit:")
    for strategy in STRATEGY_ORDER:
        d = cumulative_data.loc[cumulative_data["strategy"].eq(strategy)].copy()
        if d.empty:
            continue
        final_value = pd.to_numeric(d.sort_values("date")["cumulative_net_profit_eur"], errors="coerce").dropna().iloc[-1]
        print(f"  - {strategy}: {final_value:,.2f} EUR")
    if component_status["revenue"] or component_status["cost"]:
        print("[OK] component availability notes:")
        for message in component_status["revenue"] + component_status["cost"]:
            print(f"  - {message}")
    print(f"[OK] outputs under: {args.out_root / 'result_section'}")
    for path in outputs:
        print(f"[OK] generated: {path}")
    prune_counts = _prune_unselected_formats(args.out_root, formats)
    pruned = {suffix: count for suffix, count in prune_counts.items() if count}
    if pruned:
        print("[OK] pruned unselected RQ3 output formats: " + ", ".join(f"{suffix}={count}" for suffix, count in sorted(pruned.items())))
    if not args.skip_export:
        _export_output_tree(args.out_root, args.export_dir)
        print(f"[OK] exported RQ3 benchmark folder: {args.export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
