#!/usr/bin/env python3
"""Analyze quantile trends in trading aggressiveness from existing RQ2 outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    }
)

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is optional for the report.
    stats = None


RUN_NAME = "thesis_final_multi_2m_20260624T141002Z"
MODEL_ORDER = ["RLQR", "XGB", "TFT"]
QUANTILE_ORDER = ["p10", "p30", "p50", "p70", "p90"]
QUANTILE_LEVEL = {"p10": 0.10, "p30": 0.30, "p50": 0.50, "p70": 0.70, "p90": 0.90}

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from energy_trading.evaluation.style import THESIS_PALETTE  # noqa: E402

THESIS_ROOT = Path(
    "/Users/leori/Desktop/ uni/3 Master IS/25 MA/"
    "MA___Elevated_Energy_Trading__Forecasting_Expected_Profit_From_Participating_in_the_Day_Ahead_and_Balancing_Markets"
)
SUMMARY_CSV = (
    THESIS_ROOT
    / "figures/4-results/rq2_simulation_benchmark/backup/csv/rq2_scenario_summary_long.csv"
)
OUT_ROOT = (
    THESIS_ROOT
    / "figures/4-results/rq2_simulation_benchmark/appendix/aggressiveness_quantile_trend"
)


@dataclass(frozen=True)
class ScenarioFile:
    model: str
    model_key: str
    quantile: str
    hourly_path: Path


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _sum_abs(df: pd.DataFrame, cols: Iterable[str]) -> float:
    return float(sum(_num(df, col).abs().sum() for col in cols))


def _count_positive(df: pd.DataFrame, cols: Iterable[str]) -> int:
    if not cols:
        return 0
    total = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    for col in cols:
        total = total + _num(df, col).abs()
    return int((total > 1e-9).sum())


def _daily_risk(df: pd.DataFrame, ts: pd.Series) -> tuple[float, float]:
    pnl = _num(df, "real_pnl_eur")
    daily = pnl.groupby(ts.dt.date).sum().sort_index()
    if daily.empty:
        return np.nan, np.nan
    loss_day_share = float((daily < 0.0).mean() * 100.0)
    cumulative = daily.cumsum()
    drawdown = cumulative.cummax() - cumulative
    return loss_day_share, float(drawdown.max())


def _infer_cycle_count(df: pd.DataFrame, throughput_mwh: float) -> float:
    min_col = _num(df, "physical_soc_min_mwh")
    max_col = _num(df, "physical_soc_max_mwh")
    usable = float((max_col.replace(0.0, np.nan).max() - min_col.min()))
    if not np.isfinite(usable) or usable <= 0:
        min_col = _num(df, "real_soc_min_mwh")
        max_col = _num(df, "real_soc_max_mwh")
        usable = float((max_col.replace(0.0, np.nan).max() - min_col.min()))
    if not np.isfinite(usable) or usable <= 0:
        return np.nan
    return float(throughput_mwh / (2.0 * usable))


def discover_scenarios(summary: pd.DataFrame) -> list[ScenarioFile]:
    rows = summary[
        summary["model"].isin(MODEL_ORDER)
        & summary["quantile"].isin(QUANTILE_ORDER)
        & summary["folder"].astype(str).str.contains("_p", regex=False)
    ].copy()
    scenarios: list[ScenarioFile] = []
    for _, row in rows.iterrows():
        folder = str(row["folder"])
        quantile = str(row["quantile"])
        path = (
            REPO_ROOT
            / "artifacts/simulation_runs"
            / RUN_NAME
            / folder
            / "multi"
            / f"{quantile}_{quantile}"
            / "model_hourly.parquet"
        )
        if path.exists():
            scenarios.append(
                ScenarioFile(
                    model=str(row["model"]),
                    model_key=str(row["model_key"]),
                    quantile=quantile,
                    hourly_path=path,
                )
            )
    return sorted(
        scenarios,
        key=lambda s: (MODEL_ORDER.index(s.model), QUANTILE_ORDER.index(s.quantile)),
    )


def collect_indicators(summary: pd.DataFrame, scenarios: list[ScenarioFile]) -> pd.DataFrame:
    summary_idx = summary.set_index(["model", "quantile"])
    records: list[dict[str, float | str]] = []
    for scenario in scenarios:
        df = pd.read_parquet(scenario.hourly_path)
        ts = pd.to_datetime(
            df["timestamp_utc"] if "timestamp_utc" in df.columns else df.index,
            utc=True,
        )
        srow = summary_idx.loc[(scenario.model, scenario.quantile)]
        throughput_mwh = float(_num(df, "real_throughput_mwh").sum())
        loss_day_share, max_drawdown_eur = _daily_risk(df, ts)
        id_recourse_volume_mwh = _sum_abs(
            df,
            [
                "real_id_repair_mwh",
                "real_da_lockbook_id_repair_total_mwh",
                "real_bem_expected_activation_id_repair_total_mwh",
                "real_id_buy_mwh",
                "real_id_sell_mwh",
            ],
        )
        record = {
            "model": scenario.model,
            "model_key": scenario.model_key,
            "quantile": scenario.quantile,
            "quantile_level": QUANTILE_LEVEL[scenario.quantile],
            "source_file": str(scenario.hourly_path.relative_to(REPO_ROOT)),
            "total_net_profit_eur": float(srow["realized_profit_eur"]),
            "annualized_net_profit_eur_per_year": float(srow["annualized_profit_eur_per_year"]),
            "missed_activations_count": _count_positive(df, ["real_missed_activation_mwh"]),
            "missed_activation_mwh": float(_num(df, "real_missed_activation_mwh").sum()),
            "soc_headroom_violation_count": _count_positive(
                df,
                [
                    "real_reserve_headroom_pos_violation_mw",
                    "real_reserve_headroom_neg_violation_mw",
                    "real_headroom_violation_pos_mwh",
                    "real_headroom_violation_neg_mwh",
                    "real_protected_soc_violation_pos_mwh",
                    "real_protected_soc_violation_neg_mwh",
                    "physical_soc_violation_pos_mwh",
                    "physical_soc_violation_neg_mwh",
                    "real_power_violation_total_mw",
                ],
            ),
            "id_recourse_volume_mwh": id_recourse_volume_mwh,
            "id_recourse_cost_eur": float(abs(srow.get("id_net_revenue_eur", 0.0))),
            "terminal_soc_repair_cost_eur": float(abs(srow.get("terminal_soc_repair_cost_eur", 0.0))),
            "cleared_bcm_volume_mwh": _sum_abs(
                df,
                ["real_bcm_capacity_awarded_pos_mw", "real_bcm_capacity_awarded_neg_mw"],
            ),
            "bem_activation_volume_mwh": _sum_abs(
                df,
                ["real_bem_only_executed_pos_mwh", "real_bem_only_executed_neg_mwh"],
            ),
            "bcm_linked_bem_activation_volume_mwh": _sum_abs(
                df,
                ["real_bcm_linked_pos_activation_mwh", "real_bcm_linked_neg_activation_mwh"],
            ),
            "total_throughput_mwh": throughput_mwh,
            "cycle_count_estimate": _infer_cycle_count(df, throughput_mwh),
            "loss_day_share_percent": loss_day_share,
            "max_drawdown_eur": max_drawdown_eur,
            "penalty_cost_eur": float(abs(srow.get("penalty_cost_eur", 0.0))),
            "degradation_cost_eur": float(abs(srow.get("realized_degradation_cost_eur", 0.0))),
            "auxiliary_cost_eur": float(abs(srow.get("realized_aux_cost_eur", 0.0))),
        }
        records.append(record)
    return pd.DataFrame(records)


def add_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    score_groups = {
        "operational_feasibility_risk_score": [
            "missed_activations_count",
            "soc_headroom_violation_count",
            "penalty_cost_eur",
            "terminal_soc_repair_cost_eur",
            "id_recourse_cost_eur",
        ],
        "market_exposure_intensity_score": [
            "cleared_bcm_volume_mwh",
            "bem_activation_volume_mwh",
            "total_throughput_mwh",
            "cycle_count_estimate",
        ],
    }
    composite_indicators = sorted(
        {
            "missed_activations_count",
            "soc_headroom_violation_count",
            "id_recourse_cost_eur",
            "terminal_soc_repair_cost_eur",
            "cleared_bcm_volume_mwh",
            "bem_activation_volume_mwh",
            "total_throughput_mwh",
            "cycle_count_estimate",
            "loss_day_share_percent",
            "max_drawdown_eur",
            "penalty_cost_eur",
        }
    )
    available: dict[str, list[str]] = {"composite_aggressiveness_score": []}
    out = df.copy()
    for col in composite_indicators:
        values = pd.to_numeric(out[col], errors="coerce")
        if values.notna().sum() == 0:
            continue
        norm_col = f"{col}_rank_norm"
        if values.nunique(dropna=True) <= 1:
            out[norm_col] = 0.5
        else:
            out[norm_col] = values.rank(method="average", pct=True)
        available["composite_aggressiveness_score"].append(norm_col)
    out["composite_aggressiveness_score"] = out[
        available["composite_aggressiveness_score"]
    ].mean(axis=1)
    for score_name, cols in score_groups.items():
        norm_cols = []
        for col in cols:
            norm_col = f"{col}_rank_norm"
            if norm_col in out.columns:
                norm_cols.append(norm_col)
        available[score_name] = norm_cols
        out[score_name] = out[norm_cols].mean(axis=1)
    return out, available


def _corr(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float]:
    valid = x.notna() & y.notna()
    if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return np.nan, np.nan
    if stats is None:
        return float(x[valid].corr(y[valid], method=method)), np.nan
    if method == "pearson":
        res = stats.pearsonr(x[valid], y[valid])
    else:
        res = stats.spearmanr(x[valid], y[valid])
    return float(res.statistic), float(res.pvalue)


def trend_diagnostics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = [
        "composite_aggressiveness_score",
        "operational_feasibility_risk_score",
        "market_exposure_intensity_score",
    ]
    pooled = []
    modelwise = []
    for score in scores:
        x = df["quantile_level"]
        y = df[score]
        pr, pp = _corr(x, y, "pearson")
        sr, sp = _corr(x, y, "spearman")
        slope = float(np.polyfit(x, y, 1)[0])
        pooled.append(
            {
                "scope": "pooled",
                "score": score,
                "n": int(y.notna().sum()),
                "pearson_r": pr,
                "pearson_p": pp,
                "spearman_rho": sr,
                "spearman_p": sp,
                "linear_slope": slope,
            }
        )
        for model in MODEL_ORDER:
            sub = df[df["model"].eq(model)]
            x = sub["quantile_level"]
            y = sub[score]
            pr, pp = _corr(x, y, "pearson")
            sr, sp = _corr(x, y, "spearman")
            slope = float(np.polyfit(x, y, 1)[0])
            modelwise.append(
                {
                    "model": model,
                    "score": score,
                    "n": int(y.notna().sum()),
                    "pearson_r": pr,
                    "pearson_p": pp,
                    "spearman_rho": sr,
                    "spearman_p": sp,
                    "linear_slope": slope,
                }
            )
    return pd.DataFrame(pooled), pd.DataFrame(modelwise)


def quantile_summary(df: pd.DataFrame) -> pd.DataFrame:
    scores = [
        "composite_aggressiveness_score",
        "operational_feasibility_risk_score",
        "market_exposure_intensity_score",
    ]
    grouped = (
        df.groupby(["quantile", "quantile_level"], as_index=False)[scores]
        .agg(["mean", "median"])
        .reset_index()
    )
    grouped.columns = [
        "_".join([c for c in col if c]) if isinstance(col, tuple) else col
        for col in grouped.columns
    ]
    return grouped.sort_values("quantile_level")


def _format_float(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.{digits}f}"


def write_latex_tables(qsum: pd.DataFrame, pooled: pd.DataFrame, modelwise: pd.DataFrame) -> None:
    table_dir = OUT_ROOT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    qpath = table_dir / "quantile_aggressiveness_summary.tex"
    with qpath.open("w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
        f.write("\\caption{Aggressiveness diagnostics by quantile policy. Scores are percentile-rank normalized across model-quantile policies; higher values indicate higher exposure or risk.}\n")
        f.write("\\label{tab:rq2-quantile-aggressiveness-summary}\n")
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("\\textbf{Quantile} & \\textbf{Composite mean} & \\textbf{Composite median} & \\textbf{Feasibility mean} & \\textbf{Feasibility median} & \\textbf{Exposure mean} & \\textbf{Exposure median} \\\\\n\\midrule\n")
        for _, row in qsum.iterrows():
            f.write(
                f"{row['quantile']} & "
                f"{_format_float(row['composite_aggressiveness_score_mean'])} & "
                f"{_format_float(row['composite_aggressiveness_score_median'])} & "
                f"{_format_float(row['operational_feasibility_risk_score_mean'])} & "
                f"{_format_float(row['operational_feasibility_risk_score_median'])} & "
                f"{_format_float(row['market_exposure_intensity_score_mean'])} & "
                f"{_format_float(row['market_exposure_intensity_score_median'])} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    dpath = table_dir / "quantile_aggressiveness_trend_diagnostics.tex"
    with dpath.open("w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
        f.write("\\caption{Descriptive quantile trend diagnostics for aggressiveness scores. Correlations and slopes are descriptive because only 15 model-quantile policies are available, and only five policies per model.}\n")
        f.write("\\label{tab:rq2-quantile-aggressiveness-trend-diagnostics}\n")
        f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
        f.write("\\textbf{Scope} & \\textbf{Score} & \\textbf{$r$} & \\textbf{$\\rho$} & \\textbf{Slope} & \\textbf{$n$} \\\\\n\\midrule\n")
        combined = pd.concat(
            [
                pooled.assign(scope_label="Pooled"),
                modelwise.assign(scope_label=modelwise["model"]),
            ],
            ignore_index=True,
        )
        label_map = {
            "composite_aggressiveness_score": "Composite",
            "operational_feasibility_risk_score": "Feasibility",
            "market_exposure_intensity_score": "Exposure",
        }
        for _, row in combined.iterrows():
            f.write(
                f"{row['scope_label']} & {label_map[row['score']]} & "
                f"{_format_float(row['pearson_r'])} & {_format_float(row['spearman_rho'])} & "
                f"{_format_float(row['linear_slope'])} & {int(row['n'])} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def write_figures(df: pd.DataFrame) -> None:
    fig_dir = OUT_ROOT / "figures"
    latex_dir = OUT_ROOT / "latex_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "RLQR": THESIS_PALETTE["secondary"],
        "XGB": THESIS_PALETTE["primary"],
        "TFT": THESIS_PALETTE["tertiary"],
    }
    markers = {"RLQR": "o", "XGB": "s", "TFT": "^"}
    score_specs = [
        (
            "composite_aggressiveness_score",
            "quantile_aggressiveness_composite",
            "Composite aggressiveness score",
        ),
        (
            "operational_feasibility_risk_score",
            "quantile_aggressiveness_feasibility",
            "Operational feasibility risk score",
        ),
        (
            "market_exposure_intensity_score",
            "quantile_aggressiveness_exposure",
            "Market exposure intensity score",
        ),
    ]
    for score, stem, ylabel in score_specs:
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        for model in MODEL_ORDER:
            sub = df[df["model"].eq(model)].sort_values("quantile_level")
            ax.plot(
                sub["quantile_level"],
                sub[score],
                marker=markers[model],
                linewidth=1.8,
                label=model,
                color=colors[model],
                markersize=5.5,
            )
        ax.set_xticks([QUANTILE_LEVEL[q] for q in QUANTILE_ORDER], QUANTILE_ORDER)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Quantile policy")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False)
        fig.tight_layout()
        png_path = fig_dir / f"{stem}.png"
        fig.savefig(png_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        with (latex_dir / f"{stem}.tex").open("w") as f:
            f.write("\\begin{figure}[htbp]\n\\centering\n")
            f.write(
                f"\\includegraphics[width=0.92\\textwidth]{{figures/4-results/rq2_simulation_benchmark/appendix/aggressiveness_quantile_trend/figures/{stem}.png}}\n"
            )
            caption = {
                "composite_aggressiveness_score": "Composite aggressiveness score by quantile policy and model. Scores are percentile-rank normalized across model-quantile policies; higher values indicate higher exposure or risk.",
                "operational_feasibility_risk_score": "Operational feasibility risk score by quantile policy and model. The score combines missed activations, SoC/headroom violations, penalty costs, terminal SoC repair costs and ID recourse costs.",
                "market_exposure_intensity_score": "Market exposure intensity score by quantile policy and model. The score combines cleared BCM volume, BEM activation volume, throughput and estimated cycle count.",
            }[score]
            f.write(f"\\caption{{{caption}}}\n")
            f.write(f"\\label{{fig:rq2-{stem.replace('_', '-')}}}\n")
            f.write("\\end{figure}\n")


def write_conclusion(pooled: pd.DataFrame, modelwise: pd.DataFrame) -> None:
    text_dir = OUT_ROOT / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    comp = pooled[pooled["score"].eq("composite_aggressiveness_score")].iloc[0]
    mw = modelwise[modelwise["score"].eq("composite_aggressiveness_score")]
    signs = np.sign(mw["linear_slope"].to_numpy(dtype=float))
    if np.all(signs > 0) and comp["spearman_rho"] > 0.5:
        sentence = (
            "The aggressiveness diagnostics indicate a positive quantile trend, meaning that "
            "higher quantile policies are generally associated with higher operational exposure and risk."
        )
    elif np.all(signs < 0) and comp["spearman_rho"] < -0.5:
        sentence = (
            "The aggressiveness diagnostics indicate a negative quantile trend, meaning that "
            "lower quantile policies are generally associated with higher operational exposure and risk."
        )
    elif len(set(signs)) > 1:
        sentence = (
            "The aggressiveness diagnostics suggest that the quantile-risk relationship is "
            "model-dependent, since the trend differs across XGB, TFT and RLQR rather than "
            "following one consistent pattern."
        )
    else:
        sentence = (
            "The aggressiveness diagnostics do not indicate a consistent quantile trend, meaning "
            "that trading aggressiveness appears to be more model- and market-dependent than "
            "determined by the quantile level alone."
        )
    (text_dir / "quantile_aggressiveness_conclusion.txt").write_text(sentence + "\n")


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing summary CSV: {SUMMARY_CSV}")
    summary = pd.read_csv(SUMMARY_CSV)
    scenarios = discover_scenarios(summary)
    if len(scenarios) != 15:
        found = [(s.model, s.quantile) for s in scenarios]
        print(f"WARNING: expected 15 model-quantile files, found {len(scenarios)}: {found}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_dir = OUT_ROOT / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    indicators = collect_indicators(summary, scenarios)
    scored, available = add_scores(indicators)
    qsum = quantile_summary(scored)
    pooled, modelwise = trend_diagnostics(scored)

    scored.to_csv(csv_dir / "quantile_aggressiveness_indicators.csv", index=False)
    qsum.to_csv(csv_dir / "quantile_aggressiveness_by_quantile.csv", index=False)
    pooled.to_csv(csv_dir / "quantile_aggressiveness_pooled_correlations.csv", index=False)
    modelwise.to_csv(csv_dir / "quantile_aggressiveness_modelwise_correlations.csv", index=False)
    pd.DataFrame(
        [{"score": k, "normalized_inputs": ";".join(v)} for k, v in available.items()]
    ).to_csv(csv_dir / "quantile_aggressiveness_score_inputs.csv", index=False)

    write_latex_tables(qsum, pooled, modelwise)
    write_figures(scored)
    write_conclusion(pooled, modelwise)

    print("Wrote quantile aggressiveness diagnostics to:")
    print(OUT_ROOT)
    print("\nPooled correlations:")
    print(pooled.to_string(index=False))
    print("\nMean/median by quantile:")
    print(qsum.to_string(index=False))


if __name__ == "__main__":
    main()
