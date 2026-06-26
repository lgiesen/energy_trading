#!/usr/bin/env python3
"""Build RQ2 clearing-mechanism scatter plots from existing simulation outputs.

The script is diagnostic only. It reads completed model_hourly.parquet files and
does not rerun simulations or modify simulation artifacts.
"""

from __future__ import annotations

import argparse
import math
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

from energy_trading.evaluation.style import THESIS_PALETTE, apply_geo_style, get_model_color  # noqa: E402

DEFAULT_RUN_ROOT_CANDIDATES = (
    Path("artifacts/simulation_runs/thesis_final_multi_2m_20260624T141002Z"),
    Path("artifacts/simulation_runs/thesis_final_multi_2m_20260620T091938Z"),
)
DEFAULT_OUT_DIR = Path("artifacts/benchmark/rq2_simulation_benchmark/figures/clearing_mechanism")
DEFAULT_MODELS = ("linear", "xgb", "tft")
DEFAULT_QUANTILES = ("p10", "p30", "p50", "p70", "p90")
MODEL_LABELS = {"linear": "RLQR", "xgb": "XGB", "tft": "TFT"}
MODEL_ORDER = ("RLQR", "XGB", "TFT")


@dataclass(frozen=True)
class Scenario:
    folder: str
    model: str
    model_label: str
    quantile: str
    hourly_path: Path


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _sum_abs(df: pd.DataFrame, cols: list[str]) -> float:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required volume columns: {missing}")
    return float(sum(_num(df, c).abs().fillna(0.0).sum() for c in cols))


def _sum_signed(df: pd.DataFrame, cols: list[str]) -> float:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required revenue columns: {missing}")
    return float(sum(_num(df, c).fillna(0.0).sum() for c in cols))


def _read_hourly(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported hourly file type: {path}")


def _default_run_root() -> Path:
    for path in DEFAULT_RUN_ROOT_CANDIDATES:
        if path.exists():
            return path
    return DEFAULT_RUN_ROOT_CANDIDATES[0]


def _discover_scenarios(run_root: Path, models: tuple[str, ...], quantiles: tuple[str, ...]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for model in models:
        for quantile in quantiles:
            folder = f"{model}_{quantile}" if model != "xgb" else f"xgb_{quantile}"
            candidates = [
                run_root / folder / "multi" / f"{quantile}_{quantile}" / "model_hourly.parquet",
                run_root / folder / "multi" / f"{quantile}_{quantile}" / "model_hourly.csv",
                run_root / folder / "multi" / "model_hourly.parquet",
                run_root / folder / "multi" / "model_hourly.csv",
            ]
            hourly = next((p for p in candidates if p.exists()), None)
            if hourly is None:
                continue
            scenarios.append(
                Scenario(
                    folder=folder,
                    model=model,
                    model_label=MODEL_LABELS.get(model, model.upper()),
                    quantile=quantile,
                    hourly_path=hourly,
                )
            )
    return scenarios


def _aggregate_da(df: pd.DataFrame) -> tuple[float, float, float, str]:
    submitted = _sum_abs(df, ["da_submitted_buy_mwh", "da_submitted_sell_mwh"])
    cleared = _sum_abs(df, ["da_executed_buy_mwh", "da_executed_sell_mwh"])
    revenue = _sum_signed(df, ["real_revenue_da_eur"]) - _sum_signed(df, ["real_cost_da_eur"])
    return submitted, cleared, revenue, "EUR per cleared MWh net DA revenue"


def _aggregate_bcm(df: pd.DataFrame) -> tuple[float, float, float, str]:
    required = [
        "real_bcm_precommit_candidate_pos_mw",
        "real_bcm_precommit_candidate_neg_mw",
        "real_bcm_precommit_locked_pos_mw",
        "real_bcm_precommit_locked_neg_mw",
        "real_ev_bcm_capacity_bid_price_pos_bin_0_eur_per_mw_h",
        "real_ev_bcm_capacity_bid_price_neg_bin_0_eur_per_mw_h",
        "real_bcm_capacity_clearing_price_pos_eur_per_mw_h",
        "real_bcm_capacity_clearing_price_neg_eur_per_mw_h",
        "real_bcm_capacity_revenue_eur",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required BCM columns: {missing}")

    submitted_mwh = 0.0
    cleared_mwh = 0.0
    for side in ("pos", "neg"):
        cand = _num(df, f"real_bcm_precommit_candidate_{side}_mw").fillna(0.0).clip(lower=0.0)
        locked = _num(df, f"real_bcm_precommit_locked_{side}_mw").fillna(0.0).clip(lower=0.0)
        bid = _num(df, f"real_ev_bcm_capacity_bid_price_{side}_bin_0_eur_per_mw_h")
        clearing = _num(df, f"real_bcm_capacity_clearing_price_{side}_eur_per_mw_h")
        rejected = (cand - locked).clip(lower=0.0)
        price_rejected = rejected.where(bid.notna() & np.isfinite(bid) & bid.gt(clearing), 0.0)
        submitted_mwh += float((locked + price_rejected).abs().sum())
        cleared_mwh += float(locked.abs().sum())
    revenue = _sum_signed(df, ["real_bcm_capacity_revenue_eur"])
    return submitted_mwh, cleared_mwh, revenue, "EUR per cleared MW-h capacity"


def _aggregate_bem(df: pd.DataFrame) -> tuple[float, float, float, str]:
    submitted_cols = ["real_bem_only_submitted_pos_mw", "real_bem_only_submitted_neg_mw"]
    cleared_cols = ["real_bem_only_executed_pos_mwh", "real_bem_only_executed_neg_mwh"]
    submitted = _sum_abs(df, submitted_cols)
    cleared = _sum_abs(df, cleared_cols)
    revenue = _sum_signed(df, ["real_bem_only_activation_revenue_eur"])
    return submitted, cleared, revenue, "EUR per activated MWh"


def build_mechanism_data(run_root: Path, models: tuple[str, ...], quantiles: tuple[str, ...]) -> tuple[pd.DataFrame, list[str]]:
    aggregators = {"DA": _aggregate_da, "BCM": _aggregate_bcm, "BEM": _aggregate_bem}
    scenarios = _discover_scenarios(run_root, models, quantiles)
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    if not scenarios:
        raise FileNotFoundError(f"No model_hourly files found under {run_root}")
    for scenario in scenarios:
        df = _read_hourly(scenario.hourly_path)
        for market, aggregator in aggregators.items():
            try:
                submitted, cleared, revenue, unit = aggregator(df)
            except KeyError as exc:
                warnings.append(f"{scenario.folder} {market}: {exc}")
                continue
            if submitted <= 0.0:
                warnings.append(f"{scenario.folder} {market}: skipped because submitted volume <= 0")
                continue
            if cleared <= 0.0:
                warnings.append(f"{scenario.folder} {market}: skipped because cleared volume <= 0")
                continue
            rows.append(
                {
                    "scenario": scenario.folder,
                    "model": scenario.model,
                    "model_label": scenario.model_label,
                    "quantile": scenario.quantile,
                    "label": f"{scenario.model_label} {scenario.quantile}",
                    "market": market,
                    "submitted_volume": submitted,
                    "cleared_volume": cleared,
                    "clearing_ratio": cleared / submitted,
                    "total_revenue_eur": revenue,
                    "revenue_per_cleared_unit": revenue / cleared,
                    "revenue_per_cleared_unit_label": unit,
                    "hourly_path": str(scenario.hourly_path),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No clearing-mechanism rows could be computed. See warnings for missing columns or zero volumes.")
    return out, warnings


def _plot_market(data: pd.DataFrame, market: str, out_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    import matplotlib.pyplot as plt

    apply_geo_style()
    d = data.loc[data["market"].eq(market)].copy()
    if d.empty:
        return []
    max_submitted = float(d["submitted_volume"].max())
    sizes = 80.0 + 520.0 * (d["submitted_volume"] / max_submitted) if max_submitted > 0 else pd.Series(160.0, index=d.index)
    color_map = {"RLQR": get_model_color("linear"), "XGB": get_model_color("xgb"), "TFT": get_model_color("tft")}
    marker_map = {"RLQR": "o", "XGB": "s", "TFT": "^"}
    unit = str(d["revenue_per_cleared_unit_label"].dropna().iloc[0])
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for model in MODEL_ORDER:
        g = d.loc[d["model_label"].eq(model)].copy()
        if g.empty:
            continue
        ax.scatter(
            g["clearing_ratio"],
            g["revenue_per_cleared_unit"],
            s=sizes.loc[g.index],
            label=model,
            color=color_map.get(model, THESIS_PALETTE["neutral_dark"]),
            marker=marker_map.get(model, "o"),
            edgecolor="black",
            linewidth=0.45,
            alpha=0.88,
        )
        for _, row in g.iterrows():
            ax.annotate(
                str(row["label"]),
                (row["clearing_ratio"], row["revenue_per_cleared_unit"]),
                textcoords="offset points",
                xytext=(4, 4),
                ha="left",
                va="bottom",
                fontsize=8,
                color=THESIS_PALETTE["neutral_dark"],
            )
    ax.set_xlabel("Clearing ratio")
    ax.set_ylabel(unit)
    ax.set_title(f"{market}: clearing mechanism by model and quantile")
    ax.set_xlim(left=max(0.0, min(0.0, float(d["clearing_ratio"].min()) - 0.04)), right=min(1.05, max(1.0, float(d["clearing_ratio"].max()) + 0.04)))
    ax.grid(True, alpha=0.35)
    ax.legend(title="Model", loc="best", frameon=True)
    fig.tight_layout()
    written: list[Path] = []
    for fmt in formats:
        path = out_dir / f"clearing_mechanism_{market.lower()}.{fmt}"
        fig.savefig(path, dpi=220 if fmt == "png" else None, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def _rank_summary(data: pd.DataFrame) -> str:
    lines: list[str] = []
    for market in sorted(data["market"].unique()):
        d = data.loc[data["market"].eq(market)].copy()
        lines.append(f"{market}:")
        for col, label in [
            ("clearing_ratio", "highest clearing ratio"),
            ("revenue_per_cleared_unit", "highest revenue per cleared unit"),
            ("submitted_volume", "highest submitted volume"),
            ("total_revenue_eur", "highest total revenue"),
        ]:
            row = d.sort_values(col, ascending=False).iloc[0]
            lines.append(f"  {label}: {row['label']} ({row[col]:.4g})")
    return "\n".join(lines)


def write_readme(out_dir: Path, run_root: Path, data: pd.DataFrame, warnings: list[str], written: list[Path]) -> None:
    text = [
        "# RQ2 Clearing Mechanism Scatter Diagnostics",
        "",
        f"Input run root: `{run_root}`",
        "",
        "Markets plotted: " + ", ".join(sorted(data["market"].unique())),
        "",
        "ID is skipped because the current RQ2 outputs represent rule-based ID recourse/repairs rather than a comparable submitted-cleared auction process.",
        "",
        "BCM submitted volume is defined as awarded precommit capacity plus price-rejected precommit capacity; feasibility, EV and optimizer-filtered candidates are not treated as auction submissions.",
        "",
        "Generated files:",
        *[f"- `{p.name}`" for p in written],
        "",
        "Ranked summary:",
        "```",
        _rank_summary(data),
        "```",
    ]
    if warnings:
        text.extend(["", "Warnings:", *[f"- {w}" for w in warnings]])
    (out_dir / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build RQ2 clearing-mechanism scatter plots from existing simulation outputs.")
    ap.add_argument("--run-root", default=str(_default_run_root()))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--quantiles", default=",".join(DEFAULT_QUANTILES))
    ap.add_argument("--formats", default="png,pdf")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = tuple(x.strip() for x in str(args.models).split(",") if x.strip())
    quantiles = tuple(x.strip() for x in str(args.quantiles).split(",") if x.strip())
    formats = tuple(x.strip().lower() for x in str(args.formats).split(",") if x.strip())
    data, warnings = build_mechanism_data(run_root, models, quantiles)
    written: list[Path] = []
    for market in ("DA", "BCM", "BEM"):
        market_data = data.loc[data["market"].eq(market)].copy()
        if market_data.empty:
            warnings.append(f"{market}: no plottable rows")
            continue
        csv_path = out_dir / f"clearing_mechanism_{market.lower()}_data.csv"
        market_data.to_csv(csv_path, index=False)
        written.append(csv_path)
        written.extend(_plot_market(data, market, out_dir, formats))
    write_readme(out_dir, run_root, data, warnings, written)
    print(_rank_summary(data))
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")
    print(f"\nWrote outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
