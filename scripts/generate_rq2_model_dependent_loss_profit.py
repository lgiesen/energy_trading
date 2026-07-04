#!/usr/bin/env python3
"""Model-wise diagnostics for the RQ2 scaled MPL/profit relationship.

The script uses already exported RQ2 figure data only. It does not rerun
forecasts, simulations or metric generation.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from energy_trading.evaluation.style import THESIS_PALETTE, apply_geo_style, get_model_color  # noqa: E402

MODEL_ORDER = ["RLQR", "XGB", "TFT"]
QUANTILE_ORDER = ["p10", "p30", "p50", "p70", "p90"]
DEFAULT_INPUT = REPO_ROOT / "artifacts/benchmark/rq2_simulation_benchmark/result_section/csv/5_scaled_mean_pinball_loss_vs_net_profit.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/benchmark/rq2_simulation_benchmark"
DEFAULT_THESIS_ROOT = Path(
    "/Users/leori/Desktop/ uni/3 Master IS/25 MA/"
    "MA___Elevated_Energy_Trading__Forecasting_Expected_Profit_From_Participating_in_the_Day_Ahead_and_Balancing_Markets"
)


def _latex_escape(value: object) -> str:
    text = str(value)
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
    return "".join(repl.get(ch, ch) for ch in text)


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def _fmt_p(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    if abs(float(value)) < 0.001:
        return "<0.001"
    return f"{float(value):.3f}"


def _ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "csv": root / "appendix/csv",
        "tables": root / "appendix/tables",
        "figures": root / "appendix/figures",
        "latex_figures": root / "appendix/latex_figures",
        "text": root / "appendix/text",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _copy_outputs(paths: list[Path], output_root: Path, thesis_root: Path | None) -> None:
    if thesis_root is None:
        return
    thesis_fig_root = thesis_root / "figures/4-results/rq2_simulation_benchmark"
    for src in paths:
        rel = src.relative_to(output_root)
        dst = thesis_fig_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def load_plot_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Scaled MPL/profit input not found: {path}")
    df = pd.read_csv(path)
    required = {"model", "quantile", "scaled_mpl", "annualized_net_profit_kEUR"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input file {path} is missing required columns: {sorted(missing)}")
    out = df.rename(
        columns={
            "scaled_mpl": "scaled_mean_pinball_loss",
            "annualized_net_profit_kEUR": "annualized_net_profit_kEUR",
        }
    )[["model", "quantile", "scaled_mean_pinball_loss", "annualized_net_profit_kEUR"]].copy()
    out["model"] = out["model"].astype(str)
    out["quantile"] = out["quantile"].astype(str)
    out["model_order"] = out["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    out["quantile_order"] = out["quantile"].map({q: i for i, q in enumerate(QUANTILE_ORDER)})
    out = out.sort_values(["model_order", "quantile_order"]).drop(columns=["model_order", "quantile_order"])

    expected = {(m, q) for m in MODEL_ORDER for q in QUANTILE_ORDER}
    observed = set(zip(out["model"], out["quantile"], strict=False))
    missing_combos = sorted(expected - observed)
    if missing_combos:
        print(f"[WARN] Missing model-quantile combinations: {missing_combos}")
    if out[["scaled_mean_pinball_loss", "annualized_net_profit_kEUR"]].isna().any().any():
        bad = out[out[["scaled_mean_pinball_loss", "annualized_net_profit_kEUR"]].isna().any(axis=1)]
        raise ValueError(f"Missing plotted values in rows:\n{bad}")
    return out


def model_wise_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        g = df.loc[df["model"] == model].copy()
        x = g["scaled_mean_pinball_loss"].to_numpy(float)
        y = g["annualized_net_profit_kEUR"].to_numpy(float)
        n = len(g)
        pearson = stats.pearsonr(x, y) if n >= 3 and np.std(x) > 0 and np.std(y) > 0 else (math.nan, math.nan)
        spearman = stats.spearmanr(x, y) if n >= 3 else (math.nan, math.nan)
        lin = stats.linregress(x, y) if n >= 3 and np.std(x) > 0 else None
        if lin is None:
            slope = intercept = r_squared = slope_p = slope_ci_low = slope_ci_high = math.nan
        else:
            slope = float(lin.slope)
            intercept = float(lin.intercept)
            r_squared = float(lin.rvalue) ** 2
            slope_p = float(lin.pvalue)
            tcrit = stats.t.ppf(0.975, df=n - 2)
            slope_ci_low = slope - tcrit * float(lin.stderr)
            slope_ci_high = slope + tcrit * float(lin.stderr)
        rows.append(
            {
                "model": model,
                "n": n,
                "pearson_r": float(pearson.statistic) if hasattr(pearson, "statistic") else float(pearson[0]),
                "pearson_p": float(pearson.pvalue) if hasattr(pearson, "pvalue") else float(pearson[1]),
                "spearman_rho": float(spearman.statistic) if hasattr(spearman, "statistic") else float(spearman[0]),
                "spearman_p": float(spearman.pvalue) if hasattr(spearman, "pvalue") else float(spearman[1]),
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_squared,
                "slope_p": slope_p,
                "slope_ci_low": slope_ci_low,
                "slope_ci_high": slope_ci_high,
            }
        )
    return pd.DataFrame(rows)


def _ols(y: np.ndarray, x: np.ndarray, names: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    n, k = x.shape
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    df_resid = n - k
    ssr = float(resid @ resid)
    tss = float(((y - y.mean()) @ (y - y.mean())))
    mse = ssr / df_resid
    cov = mse * np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2 * stats.t.sf(np.abs(tvals), df_resid)
    tcrit = stats.t.ppf(0.975, df_resid)
    coef = pd.DataFrame(
        {
            "term": names,
            "coefficient": beta,
            "std_error": se,
            "t_value": tvals,
            "p_value": pvals,
            "ci_low": beta - tcrit * se,
            "ci_high": beta + tcrit * se,
        }
    )
    r2 = 1.0 - ssr / tss if tss > 0 else math.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_resid if df_resid > 0 and math.isfinite(r2) else math.nan
    summary = {"n": float(n), "k": float(k), "df_resid": float(df_resid), "ssr": ssr, "r_squared": r2, "adj_r_squared": adj_r2}
    return coef, summary


def interaction_regression(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    d = df.copy()
    d["is_xgb"] = (d["model"] == "XGB").astype(float)
    d["is_tft"] = (d["model"] == "TFT").astype(float)
    x = d["scaled_mean_pinball_loss"].to_numpy(float)
    y = d["annualized_net_profit_kEUR"].to_numpy(float)
    x_full = np.column_stack([np.ones(len(d)), x, d["is_xgb"], d["is_tft"], x * d["is_xgb"], x * d["is_tft"]])
    names_full = [
        "Intercept (RLQR)",
        "Scaled mean pinball loss",
        "XGB fixed effect",
        "TFT fixed effect",
        "Scaled mean pinball loss x XGB",
        "Scaled mean pinball loss x TFT",
    ]
    coef, full_summary = _ols(y, x_full, names_full)

    x_restricted = np.column_stack([np.ones(len(d)), x, d["is_xgb"], d["is_tft"]])
    _, restricted_summary = _ols(y, x_restricted, ["Intercept (RLQR)", "Scaled mean pinball loss", "XGB fixed effect", "TFT fixed effect"])
    df_diff = restricted_summary["df_resid"] - full_summary["df_resid"]
    ss_diff = restricted_summary["ssr"] - full_summary["ssr"]
    f_stat = (ss_diff / df_diff) / (full_summary["ssr"] / full_summary["df_resid"])
    p_value = float(stats.f.sf(f_stat, df_diff, full_summary["df_resid"]))
    nested = pd.DataFrame(
        [
            {
                "restricted_model": "profit ~ scaled MPL + model",
                "full_model": "profit ~ scaled MPL * model",
                "df_diff": df_diff,
                "ss_diff": ss_diff,
                "f_statistic": f_stat,
                "p_value": p_value,
                "interaction_improves_fit": bool(p_value < 0.05),
            }
        ]
    )
    return coef, full_summary, nested


def write_model_table(path: Path, diag: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Model-wise diagnostics for the relationship between scaled mean pinball loss and annualized net profit. Correlations and slopes are calculated separately for each model across five quantile policies. The diagnostics are interpreted descriptively because each model has only five observations.}",
        r"\label{tab:rq2-model-dependent-loss-profit-diagnostics}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Model & $n$ & Pearson $r$ & $p$ & Spearman $\rho$ & $p$ & Slope & $R^2$ \\",
        r"\midrule",
    ]
    for _, row in diag.iterrows():
        ci = f"{_fmt(row['slope_ci_low'], 1)} to {_fmt(row['slope_ci_high'], 1)}"
        lines.append(
            f"{_latex_escape(row['model'])} & {int(row['n'])} & {_fmt(row['pearson_r'])} & {_fmt_p(row['pearson_p'])} & "
            f"{_fmt(row['spearman_rho'])} & {_fmt_p(row['spearman_p'])} & {_fmt(row['slope'], 1)} & {_fmt(row['r_squared'])} \\\\"
        )
        lines.append(r"\multicolumn{8}{l}{\scriptsize Slope 95\% CI: " + ci + r"} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_interaction_table(path: Path, coef: pd.DataFrame, summary: dict[str, float]) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Interaction regression for the model-dependent loss-profit relationship. The interaction terms test whether the slope between scaled mean pinball loss and annualized net profit differs by model. Results are interpreted as exploratory diagnostics due to the small number of model-quantile observations.}",
        r"\label{tab:rq2-model-dependent-loss-profit-interaction}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Term & Coef. & Std. err. & $t$ & $p$ \\",
        r"\midrule",
    ]
    for _, row in coef.iterrows():
        lines.append(
            f"{_latex_escape(row['term'])} & {_fmt(row['coefficient'], 1)} & {_fmt(row['std_error'], 1)} & "
            f"{_fmt(row['t_value'])} & {_fmt_p(row['p_value'])} \\\\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{5}{l}{\scriptsize $R^2="
        + _fmt(summary["r_squared"])
        + r"$, adjusted $R^2="
        + _fmt(summary["adj_r_squared"])
        + r"$, $n="
        + str(int(summary["n"]))
        + r"$.} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_nested_table(path: Path, nested: pd.DataFrame) -> None:
    row = nested.iloc[0]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Nested-model comparison for the model-dependent loss-profit relationship. The comparison tests whether model-specific slope interactions improve fit beyond a common slope and model fixed effects; results are descriptive because only 15 model-quantile observations are available.}",
        r"\label{tab:rq2-model-dependent-loss-profit-nested}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Comparison & $\Delta df$ & $F$ & $p$ \\",
        r"\midrule",
        f"Add model-specific slopes & {int(row['df_diff'])} & {_fmt(row['f_statistic'])} & {_fmt_p(row['p_value'])} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(df: pd.DataFrame, png_path: Path, tex_path: Path) -> None:
    apply_geo_style()
    colors = {"RLQR": get_model_color("linear"), "XGB": get_model_color("xgb"), "TFT": get_model_color("tft")}
    markers = {"RLQR": "o", "XGB": "s", "TFT": "^"}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for model in MODEL_ORDER:
        g = df.loc[df["model"] == model].sort_values("scaled_mean_pinball_loss")
        x = g["scaled_mean_pinball_loss"].to_numpy(float)
        y = g["annualized_net_profit_kEUR"].to_numpy(float)
        ax.scatter(x, y, s=48, marker=markers[model], color=colors[model], label=model, zorder=3)
        if len(g) >= 2 and np.std(x) > 0:
            lin = stats.linregress(x, y)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, lin.intercept + lin.slope * xs, color=colors[model], linewidth=1.4, alpha=0.75)
        for _, row in g.iterrows():
            ax.annotate(
                str(row["quantile"]),
                (row["scaled_mean_pinball_loss"], row["annualized_net_profit_kEUR"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color=THESIS_PALETTE["neutral_dark"],
            )
    ax.set_xlabel("Scaled mean pinball loss")
    ax.set_ylabel("Annualized net profit (kEUR/year)")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(loc="upper right", frameon=False, ncol=3, columnspacing=1.1)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    rel_png = "figures/4-results/rq2_simulation_benchmark/appendix/figures/" + png_path.name
    tex = "\n".join(
        [
            r"\begin{figure}[htbp]",
            r"\centering",
            r"\includegraphics[width=0.86\linewidth]{" + rel_png + r"}",
            r"\caption{Model-wise descriptive loss-profit relationship by model and quantile policy. Points show scaled mean pinball loss and annualized net profit for one model-quantile policy; lines are separate linear fits by model. The fits are descriptive because each model contributes only five quantile policies.}",
            r"\label{fig:rq2-model-dependent-loss-profit}",
            r"\end{figure}",
            "",
        ]
    )
    tex_path.write_text(tex, encoding="utf-8")


def write_interpretation(path: Path, diag: pd.DataFrame, coef: pd.DataFrame, nested: pd.DataFrame) -> None:
    by_model = {row["model"]: row for _, row in diag.iterrows()}
    xgb = by_model["XGB"]
    rlqr = by_model["RLQR"]
    tft = by_model["TFT"]
    interaction_rows = coef[coef["term"].str.contains(" x ")]
    interaction_text = "; ".join(f"{r.term}: p={_fmt_p(r.p_value)}" for r in interaction_rows.itertuples())
    nested_row = nested.iloc[0]
    text = (
        "The model-wise diagnostics do not confirm a uniform negative loss-profit relationship within all "
        "model classes. In the current data, TFT shows the clearest negative descriptive association, with "
        f"Pearson r={_fmt(tft['pearson_r'])} and slope {_fmt(tft['slope'], 1)}, while XGB is nearly flat "
        f"(r={_fmt(xgb['pearson_r'])}, slope {_fmt(xgb['slope'], 1)}) and RLQR is also essentially flat "
        f"(r={_fmt(rlqr['pearson_r'])}, slope {_fmt(rlqr['slope'], 1)}). The pooled interaction regression "
        "tests whether these slopes differ "
        f"from the RLQR reference slope; interaction p-values are {interaction_text}. The nested comparison "
        f"for adding model-specific slopes gives F={_fmt(nested_row['f_statistic'])}, p={_fmt_p(nested_row['p_value'])}. "
        "Because the analysis contains only 15 observations and only five quantile policies per model, these "
        "results should be interpreted as descriptive evidence about model dependence, not as statistically "
        "robust or causal evidence."
    )
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--thesis-root", type=Path, default=DEFAULT_THESIS_ROOT)
    args = parser.parse_args()

    df = load_plot_data(args.input)
    dirs = _ensure_dirs(args.output_root)
    diag = model_wise_diagnostics(df)
    coef, summary, nested = interaction_regression(df)

    outputs: list[Path] = []
    data_path = dirs["csv"] / "model_dependent_loss_profit_plot_data.csv"
    diag_csv = dirs["csv"] / "model_dependent_loss_profit_diagnostics.csv"
    coef_csv = dirs["csv"] / "model_dependent_loss_profit_interaction_coefficients.csv"
    nested_csv = dirs["csv"] / "model_dependent_loss_profit_nested_model_comparison.csv"
    df.to_csv(data_path, index=False)
    diag.to_csv(diag_csv, index=False)
    coef.to_csv(coef_csv, index=False)
    nested.to_csv(nested_csv, index=False)
    outputs += [data_path, diag_csv, coef_csv, nested_csv]

    diag_tex = dirs["tables"] / "model_dependent_loss_profit_diagnostics.tex"
    coef_tex = dirs["tables"] / "model_dependent_loss_profit_interaction_coefficients.tex"
    nested_tex = dirs["tables"] / "model_dependent_loss_profit_nested_model_comparison.tex"
    write_model_table(diag_tex, diag)
    write_interaction_table(coef_tex, coef, summary)
    write_nested_table(nested_tex, nested)
    outputs += [diag_tex, coef_tex, nested_tex]

    png_path = dirs["figures"] / "model_dependent_loss_profit_by_model.png"
    tex_path = dirs["latex_figures"] / "model_dependent_loss_profit_by_model.tex"
    write_figure(df, png_path, tex_path)
    outputs += [png_path, tex_path]

    interpretation = dirs["text"] / "model_dependent_loss_profit_interpretation.txt"
    write_interpretation(interpretation, diag, coef, nested)
    outputs.append(interpretation)

    _copy_outputs(outputs, args.output_root, args.thesis_root)

    print("\nModel-wise diagnostics:")
    print(diag.to_string(index=False))
    print("\nInteraction regression coefficients:")
    print(coef.to_string(index=False))
    print("\nNested model comparison:")
    print(nested.to_string(index=False))
    print(f"\n[OK] Wrote {len(outputs)} outputs under {args.output_root / 'appendix'}")
    if args.thesis_root:
        print(f"[OK] Copied outputs to {args.thesis_root / 'figures/4-results/rq2_simulation_benchmark/appendix'}")


if __name__ == "__main__":
    main()
