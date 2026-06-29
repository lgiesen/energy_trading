#!/usr/bin/env python3
"""Generate concise RQ2 downside and operational risk diagnostics.

This script is a reporting layer only. It reads existing RQ2 benchmark and
simulation outputs and does not run simulations or modify simulation artifacts.
"""

from __future__ import annotations

import argparse
import shutil
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from energy_trading.evaluation.style import GEO_SEQUENTIAL_BLUE, THESIS_PALETTE, apply_geo_style


DEFAULT_RQ2_ROOT = Path("artifacts/benchmark/rq2_simulation_benchmark")
DEFAULT_RUN_ROOT = Path("artifacts/simulation_runs/thesis_final_multi_2m_20260624T141002Z")
MODEL_ORDER = ["RLQR", "XGB", "TFT"]
QUANTILE_ORDER = ["p10", "p30", "p50", "p70", "p90"]
BENCHMARK_ORDER = ["Naive", "RHPF"]


@dataclass(frozen=True)
class MetricSpec:
    column: str
    filename: str
    label: str
    unit: str
    formatter: str


HEATMAP_SPECS = [
    MetricSpec("loss_day_share", "risk_heatmap_loss_day_share", "Loss-day share", "%", "percent"),
    MetricSpec("max_drawdown_eur", "risk_heatmap_max_drawdown", "Max drawdown", "kEUR", "keur_abs"),
    MetricSpec("daily_pnl_cvar_5_eur", "risk_heatmap_cvar_5", "Daily PnL CVaR 5%", "kEUR", "keur"),
    MetricSpec("profit_volatility_eur", "risk_heatmap_profit_volatility", "Profit volatility", "kEUR", "keur"),
    MetricSpec("fallback_share", "risk_heatmap_fallback", "Fallback share", "%", "percent"),
    MetricSpec("missed_activation_count", "risk_heatmap_missed_activation", "Missed activations", "count", "count"),
    MetricSpec("soc_headroom_violation_count", "risk_heatmap_soc_headroom_violations", "SoC/headroom violations", "count", "count"),
]


def _safe_num(value: Any) -> float:
    x = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(x) if pd.notna(x) and math.isfinite(float(x)) else math.nan


def _ensure_dirs(rq2_root: Path) -> dict[str, Path]:
    dirs = {
        "result_csv": rq2_root / "result_section" / "csv",
        "result_tables": rq2_root / "result_section" / "tables",
        "result_latex_tables": rq2_root / "result_section" / "latex_tables",
        "appendix_figures": rq2_root / "appendix" / "figures" / "risk_diagnostics",
        "appendix_csv": rq2_root / "appendix" / "csv" / "risk_diagnostics",
        "warnings": rq2_root / "backup" / "warnings",
        "diagnostics": rq2_root / "backup" / "diagnostics",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def _find_hourly_path(run_root: Path, folder: str, scenario: str) -> Path | None:
    folder_path = run_root / str(folder)
    candidates = [
        folder_path / "multi" / str(scenario) / "model_hourly.parquet",
        folder_path / "multi" / str(scenario) / "backtest_hourly.parquet",
        folder_path / str(scenario) / "model_hourly.parquet",
        folder_path / str(scenario) / "backtest_hourly.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(folder_path.glob("**/model_hourly.parquet")) + sorted(folder_path.glob("**/backtest_hourly.parquet"))
    return matches[0] if matches else None


def _daily_pnl_from_hourly(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    if "timestamp_utc" not in df.columns:
        raise ValueError(f"Missing timestamp_utc in hourly file: {path}")
    pnl_col = next((c for c in ["real_pnl_eur", "pnl_eur", "realized_pnl_eur"] if c in df.columns), None)
    if pnl_col is None:
        raise ValueError(f"Missing hourly PnL column in {path}; expected one of real_pnl_eur, pnl_eur, realized_pnl_eur")
    ts = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    pnl = pd.to_numeric(df[pnl_col], errors="coerce")
    d = pd.DataFrame({"timestamp_utc": ts, "pnl_eur": pnl}).dropna(subset=["timestamp_utc", "pnl_eur"])
    if d.empty:
        return pd.Series(dtype=float)
    d["date"] = d["timestamp_utc"].dt.floor("D")
    return d.groupby("date")["pnl_eur"].sum().sort_index()


def _financial_downside(daily_pnl: pd.Series) -> dict[str, float]:
    dvals = pd.to_numeric(daily_pnl, errors="coerce").dropna()
    if dvals.empty:
        return {
            "loss_day_share": math.nan,
            "worst_day_eur": math.nan,
            "worst_week_eur": math.nan,
            "max_drawdown_eur": math.nan,
            "daily_pnl_var_5_eur": math.nan,
            "daily_pnl_cvar_5_eur": math.nan,
            "profit_volatility_eur": math.nan,
        }
    q05 = float(dvals.quantile(0.05))
    cum = dvals.cumsum()
    drawdown = cum - cum.cummax()
    daily_df = pd.DataFrame({"date": dvals.index, "daily_pnl_eur": dvals.to_numpy(dtype=float)})
    daily_df["week"] = pd.to_datetime(daily_df["date"], utc=True).dt.tz_localize(None).dt.to_period("W").astype(str)
    return {
        "loss_day_share": float((dvals < 0.0).mean()),
        "worst_day_eur": float(dvals.min()),
        "worst_week_eur": float(daily_df.groupby("week")["daily_pnl_eur"].sum().min()),
        "max_drawdown_eur": float(drawdown.min()),
        "daily_pnl_var_5_eur": q05,
        "daily_pnl_cvar_5_eur": float(dvals[dvals <= q05].mean()) if (dvals <= q05).any() else q05,
        "profit_volatility_eur": float(dvals.std(ddof=0)),
    }


def _model_key(label: str) -> str:
    label = str(label)
    if label == "RLQR":
        return "linear"
    return label.lower()


def _build_risk_frame(rq2_root: Path, run_root: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    summary_path = rq2_root / "backup" / "csv" / "rq2_scenario_summary_long.csv"
    severity_path = rq2_root / "backup" / "diagnostics" / "simulation_invalidity_severity_summary.csv"
    summary = _read_csv_required(summary_path)
    severity = _read_csv_required(severity_path) if severity_path.exists() else pd.DataFrame()
    input_files = [str(summary_path)]
    if severity_path.exists():
        input_files.append(str(severity_path))

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for _, row in summary.iterrows():
        model = str(row.get("model", "")).strip()
        quantile = str(row.get("quantile", "")).strip()
        is_benchmark = bool(row.get("is_benchmark", False))
        if not is_benchmark and model not in MODEL_ORDER:
            continue
        if not is_benchmark and quantile not in QUANTILE_ORDER:
            continue

        scenario = str(row.get("folder", "")).strip()
        strategy = model if is_benchmark else f"{model} {quantile}"
        hourly_path = _find_hourly_path(run_root, scenario, str(row.get("scenario", "")))
        fin: dict[str, float]
        if hourly_path is None:
            warnings.append(f"{scenario}: missing hourly file; downside metrics unavailable")
            fin = _financial_downside(pd.Series(dtype=float))
        else:
            input_files.append(str(hourly_path))
            try:
                fin = _financial_downside(_daily_pnl_from_hourly(hourly_path))
            except Exception as exc:
                warnings.append(f"{scenario}: failed to compute daily downside metrics from {hourly_path}: {exc}")
                fin = _financial_downside(pd.Series(dtype=float))

        annualized_profit = _safe_num(row.get("annualized_profit_eur_per_year"))
        out = {
            "scenario": scenario,
            "strategy": strategy,
            "model": model,
            "quantile": quantile,
            "is_benchmark": is_benchmark,
            "annualized_net_profit_eur": annualized_profit,
            **fin,
            "fallback_count": math.nan,
            "fallback_share": math.nan,
            "infeasibility_count": math.nan,
            "infeasibility_share": math.nan,
            "missed_activation_count": math.nan,
            "soc_violation_count": math.nan,
            "headroom_violation_count": math.nan,
            "soc_headroom_violation_count": math.nan,
            "invalidity_count": math.nan,
            "invalidity_share": math.nan,
        }
        if not severity.empty:
            sev = severity.loc[severity["scenario"].astype(str).eq(scenario)]
            if not sev.empty:
                s = sev.iloc[0]
                fallback_count = _safe_num(s.get("fallback_optimization_count"))
                fallback_share = _safe_num(s.get("fallback_optimization_share"))
                inf_count = _safe_num(s.get("combined_infeasibility_hours"))
                inf_share = _safe_num(s.get("combined_infeasibility_hours_share"))
                missed_count = _safe_num(s.get("missed_activation_count"))
                soc_count = _safe_num(s.get("soc_violation_hours"))
                headroom_count = _safe_num(s.get("reserve_headroom_shortfall_hours"))
                out.update(
                    {
                        "fallback_count": fallback_count,
                        "fallback_share": fallback_share,
                        "infeasibility_count": inf_count,
                        "infeasibility_share": inf_share,
                        "missed_activation_count": missed_count,
                        "soc_violation_count": soc_count,
                        "headroom_violation_count": headroom_count,
                        "soc_headroom_violation_count": (0.0 if math.isnan(soc_count) else soc_count)
                        + (0.0 if math.isnan(headroom_count) else headroom_count),
                        "invalidity_count": inf_count,
                        "invalidity_share": inf_share,
                    }
                )
        rows.append(out)
    if not severity.empty:
        missing_severity = set(summary["folder"].astype(str)) - set(severity["scenario"].astype(str))
        for scenario in sorted(missing_severity):
            warnings.append(f"{scenario}: no invalidity severity row available")
    return pd.DataFrame(rows), warnings, sorted(set(input_files))


def _select_best_quantiles(risk: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    non_benchmark = risk.loc[~risk["is_benchmark"].astype(bool)].copy()
    for model in MODEL_ORDER:
        g = non_benchmark.loc[non_benchmark["model"].eq(model)].copy()
        g["annualized_net_profit_eur"] = pd.to_numeric(g["annualized_net_profit_eur"], errors="coerce")
        g = g.dropna(subset=["annualized_net_profit_eur"])
        if not g.empty:
            rows.append(g.sort_values("annualized_net_profit_eur", ascending=False).head(1))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fmt_eur_k(value: Any) -> str:
    x = _safe_num(value)
    if not math.isfinite(x):
        return "--"
    return f"{x / 1000.0:,.1f}"


def _fmt_pct(value: Any) -> str:
    x = _safe_num(value)
    if not math.isfinite(x):
        return "--"
    return f"{100.0 * x:.1f}"


def _fmt_count(value: Any) -> str:
    x = _safe_num(value)
    if not math.isfinite(x):
        return "--"
    return f"{x:,.0f}"


def _latex_escape(value: Any) -> str:
    s = str(value)
    repl = {
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
    for old, new in repl.items():
        s = s.replace(old, new)
    return s


def _write_best_table(best: pd.DataFrame, csv_path: Path, tex_path: Path) -> None:
    table = pd.DataFrame(
        {
            "Strategy": best["strategy"].astype(str),
            "Annualized net profit (kEUR)": best["annualized_net_profit_eur"].map(_fmt_eur_k),
            "Loss-day share (%)": best["loss_day_share"].map(_fmt_pct),
            "Worst day (kEUR)": best["worst_day_eur"].map(_fmt_eur_k),
            "Max drawdown (kEUR)": best["max_drawdown_eur"].map(_fmt_eur_k),
            "Daily PnL CVaR 5% (kEUR)": best["daily_pnl_cvar_5_eur"].map(_fmt_eur_k),
            "Profit volatility (kEUR)": best["profit_volatility_eur"].map(_fmt_eur_k),
            "Fallback share (%)": best["fallback_share"].map(_fmt_pct),
            "Missed activations": best["missed_activation_count"].map(_fmt_count),
            "SoC/headroom violations": best["soc_headroom_violation_count"].map(_fmt_count),
        }
    )
    table.to_csv(csv_path, index=False)

    strategies = table["Strategy"].astype(str).tolist()
    metric_columns = [col for col in table.columns if col != "Strategy"]
    align = "l" + "r" * len(strategies)
    header = [r"\textbf{Metric}", *[rf"\textbf{{{_latex_escape(strategy)}}}" for strategy in strategies]]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Downside and operational risk diagnostics for the best-performing quantile of each model. Financial values are reported in thousand EUR.}",
        r"\label{tab:risk_diagnostics_best_quantile}",
        r"\begin{tabular}{" + align + r"}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for metric in metric_columns:
        row_values = [_latex_escape(metric)]
        row_values.extend(_latex_escape(value) for value in table[metric].tolist())
        lines.append(" & ".join(row_values) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_heatmap(pivot: pd.DataFrame, spec: MetricSpec, out_base: Path) -> None:
    apply_geo_style()
    values = pivot.to_numpy(dtype=float)
    mask = np.isfinite(values)
    if not mask.any():
        raise ValueError(f"No finite values for heatmap metric {spec.column}")

    if spec.formatter in {"keur", "keur_abs"}:
        display = values / 1000.0
        color_values = np.abs(display) if spec.formatter == "keur_abs" else display
    elif spec.formatter == "percent":
        display = values * 100.0
        color_values = display
    else:
        display = values
        color_values = display

    cmap = LinearSegmentedColormap.from_list("rq2_risk_blue", [GEO_SEQUENTIAL_BLUE[f"seq_{i}"] for i in range(1, 8)])
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    im = ax.imshow(color_values, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=list(pivot.columns))
    ax.set_yticks(np.arange(len(pivot.index)), labels=list(pivot.index))
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Model")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{spec.label} ({spec.unit})")
    for yi in range(display.shape[0]):
        for xi in range(display.shape[1]):
            value = display[yi, xi]
            if not math.isfinite(float(value)):
                txt = "n/a"
            elif spec.formatter == "percent":
                txt = f"{value:.1f}"
            elif spec.formatter in {"keur", "keur_abs"}:
                txt = f"{value:.1f}"
            else:
                txt = f"{value:.0f}"
            ax.text(xi, yi, txt, ha="center", va="center", fontsize=9, color=THESIS_PALETTE["neutral_dark"])
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=220)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def _write_heatmaps(risk: pd.DataFrame, fig_dir: Path, csv_dir: Path) -> tuple[list[str], list[str]]:
    available: list[str] = []
    omitted: list[str] = []
    d = risk.loc[~risk["is_benchmark"].astype(bool)].copy()
    for spec in HEATMAP_SPECS:
        if spec.column not in d.columns:
            omitted.append(f"{spec.column}: column unavailable")
            continue
        finite = pd.to_numeric(d[spec.column], errors="coerce").dropna()
        if finite.empty:
            omitted.append(f"{spec.column}: no finite values")
            continue
        pivot = (
            d.pivot_table(index="model", columns="quantile", values=spec.column, aggfunc="first")
            .reindex(index=MODEL_ORDER, columns=QUANTILE_ORDER)
        )
        pivot.to_csv(csv_dir / f"{spec.filename}_data.csv")
        _plot_heatmap(pivot, spec, fig_dir / spec.filename)
        available.append(spec.column)
    return available, omitted


def _write_log(path: Path, *, input_files: list[str], n_strategies: int, available: list[str], omitted: list[str], warnings: list[str]) -> None:
    lines = [
        "# RQ2 Risk Diagnostics Generation Log",
        "",
        f"Strategies processed: {n_strategies}",
        "",
        "## Input files used",
        *[f"- {p}" for p in input_files],
        "",
        "## Metrics available",
        *[f"- {m}" for m in available],
        "",
        "## Metrics omitted",
        *([f"- {m}" for m in omitted] if omitted else ["- none"]),
        "",
        "## Warnings",
        *([f"- {w}" for w in warnings] if warnings else ["- none"]),
        "",
        "Interpretation note: Downside and operational risk diagnostics are descriptive. They are not statistically robust financial risk estimates because the RQ2 test period is limited.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_outputs(rq2_root: Path, run_root: Path) -> dict[str, Any]:
    dirs = _ensure_dirs(rq2_root)
    risk, warnings, input_files = _build_risk_frame(rq2_root, run_root)
    if risk.empty:
        raise RuntimeError("No RQ2 strategies available for risk diagnostics.")

    risk.to_csv(dirs["appendix_csv"] / "risk_diagnostics_all_strategies.csv", index=False)
    best = _select_best_quantiles(risk)
    if best.empty:
        raise RuntimeError("Could not select best model quantiles for risk diagnostics table.")
    _write_best_table(
        best,
        dirs["result_csv"] / "risk_diagnostics_best_quantile.csv",
        dirs["result_tables"] / "risk_diagnostics_best_quantile.tex",
    )
    shutil.copy2(
        dirs["result_tables"] / "risk_diagnostics_best_quantile.tex",
        dirs["result_latex_tables"] / "risk_diagnostics_best_quantile.tex",
    )
    available, omitted = _write_heatmaps(risk, dirs["appendix_figures"], dirs["appendix_csv"])
    _write_log(
        dirs["diagnostics"] / "risk_diagnostics_generation_log.md",
        input_files=input_files,
        n_strategies=len(risk),
        available=available,
        omitted=omitted,
        warnings=warnings,
    )
    if warnings:
        pd.DataFrame({"warning": warnings}).to_csv(dirs["warnings"] / "risk_diagnostics_warnings.csv", index=False)
    return {
        "n_strategies": len(risk),
        "available_metrics": available,
        "omitted_metrics": omitted,
        "best_csv": str(dirs["result_csv"] / "risk_diagnostics_best_quantile.csv"),
        "best_tex": str(dirs["result_tables"] / "risk_diagnostics_best_quantile.tex"),
        "best_tex_legacy": str(dirs["result_latex_tables"] / "risk_diagnostics_best_quantile.tex"),
        "heatmap_dir": str(dirs["appendix_figures"]),
        "log": str(dirs["diagnostics"] / "risk_diagnostics_generation_log.md"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rq2-root", type=Path, default=DEFAULT_RQ2_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = build_outputs(rq2_root=args.rq2_root, run_root=args.run_root)
    print(f"[OK] strategies processed: {outputs['n_strategies']}")
    print(f"[OK] best table CSV: {outputs['best_csv']}")
    print(f"[OK] best table LaTeX: {outputs['best_tex']}")
    print(f"[OK] heatmap dir: {outputs['heatmap_dir']}")
    print(f"[OK] metrics available: {', '.join(outputs['available_metrics'])}")
    if outputs["omitted_metrics"]:
        print(f"[WARN] metrics omitted: {'; '.join(outputs['omitted_metrics'])}")
    print(f"[OK] log: {outputs['log']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
