#!/usr/bin/env python3
"""Generate Relative Value of Forecast heatmap for the RQ2 benchmark."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

from energy_trading.evaluation.style import GEO_SEQUENTIAL_BLUE, THESIS_PALETTE, apply_geo_style


DEFAULT_RQ2_ROOT = Path("artifacts/benchmark/rq2_simulation_benchmark")
MODEL_ORDER = ["RLQR", "XGB", "TFT"]
QUANTILE_ORDER = ["p10", "p30", "p50", "p70", "p90"]
CAPTION = (
    "Relative VoF reports the share of the naive-to-RHPF profit gap recovered by each model-quantile policy. "
    "A value of 0\\% corresponds to the naive benchmark, 100\\% corresponds to RHPF, and higher values indicate "
    "greater realized economic value from the forecast."
)


def _safe_float(value: Any) -> float:
    x = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(x) if pd.notna(x) and math.isfinite(float(x)) else math.nan


def _tex_float(value: float, digits: int = 3) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


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


def _rq2_latex_color_defs() -> list[str]:
    return [
        r"\definecolor{rqTwoNeutral}{rgb}{0.2000,0.2000,0.2000}",
        r"\definecolor{rqTwoGrid}{rgb}{0.8471,0.8471,0.8471}",
    ]


def _axis_common_options() -> list[str]:
    return [
        r"tick align=outside,",
        r"axis line style={rqTwoNeutral},",
        r"tick style={rqTwoNeutral},",
        r"label style={font=\small},",
        r"tick label style={font=\small},",
        r"title style={font=\normalfont\small},",
        r"legend style={font=\small, draw=none, fill=none},",
        r"grid=major,",
        r"grid style={rqTwoGrid!55, line width=0.2pt},",
    ]


def _read_profit_source(rq2_root: Path) -> pd.DataFrame:
    candidates = [
        rq2_root / "backup" / "csv" / "2_quantile_sweep_net_profit_by_model.csv",
        rq2_root / "backup" / "csv" / "1_profit_heatmap.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if path.name == "2_quantile_sweep_net_profit_by_model.csv":
                required = {"model", "quantile", "annualized_net_profit_eur_per_year"}
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
                return df[["model", "quantile", "annualized_net_profit_eur_per_year"]].rename(
                    columns={"annualized_net_profit_eur_per_year": "annualized_net_profit_eur"}
                )
            required = {"quantile", "Naive", "RHPF", *MODEL_ORDER}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
            rows: list[dict[str, Any]] = []
            for _, row in df.iterrows():
                for model in ["Naive", "RHPF", *MODEL_ORDER]:
                    rows.append(
                        {
                            "model": model,
                            "quantile": str(row["quantile"]),
                            "annualized_net_profit_eur": _safe_float(row[model]),
                        }
                    )
            return pd.DataFrame(rows)
    raise FileNotFoundError(f"Missing RQ2 net-profit source. Tried: {', '.join(str(p) for p in candidates)}")


def build_relative_vof(rq2_root: Path) -> tuple[pd.DataFrame, list[str]]:
    profits = _read_profit_source(rq2_root)
    warnings: list[str] = []
    profits["model"] = profits["model"].astype(str)
    profits["quantile"] = profits["quantile"].astype(str)
    profits["annualized_net_profit_eur"] = pd.to_numeric(profits["annualized_net_profit_eur"], errors="coerce")

    naive = profits.loc[profits["model"].eq("Naive"), "annualized_net_profit_eur"].dropna()
    rhpf = profits.loc[profits["model"].eq("RHPF"), "annualized_net_profit_eur"].dropna()
    if naive.empty:
        raise ValueError("Cannot compute Relative Value of Forecast: Naive benchmark annualized profit is missing.")
    if rhpf.empty:
        raise ValueError("Cannot compute Relative Value of Forecast: RHPF benchmark annualized profit is missing.")
    j_naive = float(naive.iloc[0])
    j_rhpf = float(rhpf.iloc[0])
    denom = j_rhpf - j_naive
    if not math.isfinite(denom) or denom <= 0.0:
        warnings.append(
            f"Relative Value of Forecast denominator is non-positive: RHPF={j_rhpf:.6g}, Naive={j_naive:.6g}, denominator={denom:.6g}."
        )

    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for quantile in QUANTILE_ORDER:
            d = profits.loc[profits["model"].eq(model) & profits["quantile"].eq(quantile)]
            if d.empty:
                value = math.nan
                j_model = math.nan
            else:
                j_model = float(d["annualized_net_profit_eur"].iloc[0])
                value = (j_model - j_naive) / denom if math.isfinite(denom) and abs(denom) > 1e-12 else math.nan
            rows.append(
                {
                    "model": model,
                    "quantile": quantile,
                    "annualized_model_net_profit_eur": j_model,
                    "annualized_naive_net_profit_eur": j_naive,
                    "annualized_rhpf_net_profit_eur": j_rhpf,
                    "relative_value_of_forecast": value,
                }
            )
    out = pd.DataFrame(rows)
    values = pd.to_numeric(out["relative_value_of_forecast"], errors="coerce")
    below = int((values < 0.0).sum())
    above = int((values > 1.0).sum())
    if below or above:
        warnings.append(f"Relative Value of Forecast outside [0, 1]: below_0={below}, above_1={above}.")
    return out, warnings


def _wide_values(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.pivot_table(index="model", columns="quantile", values="relative_value_of_forecast", aggfunc="first")
        .reindex(index=MODEL_ORDER, columns=QUANTILE_ORDER)
    )


def write_csv(data: pd.DataFrame, path: Path) -> None:
    wide = _wide_values(data).reset_index().rename(columns={"model": "model"})
    path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(path, index=False)


def plot_heatmap(data: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    apply_geo_style()
    pivot = _wide_values(data)
    values = pivot.to_numpy(dtype=float)
    color_values = np.clip(values, 0.0, 1.0)
    cmap = LinearSegmentedColormap.from_list(
        "rq2_vof_blue", [GEO_SEQUENTIAL_BLUE[f"seq_{i}"] for i in range(1, 8)]
    )
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    im = ax.imshow(color_values, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(QUANTILE_ORDER)), labels=QUANTILE_ORDER)
    ax.set_yticks(np.arange(len(MODEL_ORDER)), labels=MODEL_ORDER)
    ax.set_xlabel("Quantile policy")
    ax.set_ylabel("Model")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relative Value of Forecast")
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    for yi, model in enumerate(MODEL_ORDER):
        for xi, quantile in enumerate(QUANTILE_ORDER):
            value = values[yi, xi]
            txt = "n/a" if not math.isfinite(float(value)) else f"{float(value) * 100.0:.0f}%"
            text_color = "white" if math.isfinite(float(value)) and np.clip(value, 0.0, 1.0) >= 0.55 else THESIS_PALETTE["neutral_dark"]
            ax.text(xi, yi, txt, ha="center", va="center", fontsize=9, color=text_color)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def write_latex(data: pd.DataFrame, path: Path) -> None:
    pivot = _wide_values(data)
    palette = [GEO_SEQUENTIAL_BLUE[f"seq_{i}"] for i in range(1, 8)]

    def cell_color(value: float) -> str:
        if not math.isfinite(value):
            return "#F2F2F2"
        clipped = max(0.0, min(1.0, value))
        idx = int(round(clipped * (len(palette) - 1)))
        idx = max(0, min(len(palette) - 1, idx))
        return palette[idx]

    cells: list[str] = []
    for yi, model in enumerate(MODEL_ORDER):
        for xi, quantile in enumerate(QUANTILE_ORDER):
            value = _safe_float(pivot.loc[model, quantile]) if model in pivot.index and quantile in pivot.columns else math.nan
            txt = "n/a" if not math.isfinite(value) else f"{value * 100.0:.0f}\\%"
            color = cell_color(value)
            text_color = "white" if math.isfinite(value) and max(0.0, min(1.0, value)) >= 0.55 else "rqTwoNeutral"
            cells.extend(
                [
                    rf"\definecolor{{rqTwoVoFCell{xi}{yi}}}{{HTML}}{{{color.lstrip('#')}}}",
                    rf"\filldraw[fill=rqTwoVoFCell{xi}{yi}, draw=white, line width=0.6pt] (axis cs:{xi - 0.5},{yi - 0.5}) rectangle (axis cs:{xi + 0.5},{yi + 0.5});",
                    rf"\node[text={text_color}, font=\scriptsize] at (axis cs:{xi},{yi}) {{{txt}}};",
                ]
            )

    lines = [
        r"% Requires \usepackage{pgfplots}",
        r"% Requires \pgfplotsset{compat=1.18}",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        *_rq2_latex_color_defs(),
        r"\begin{axis}[",
        *_axis_common_options(),
        r"width=0.78\linewidth,",
        r"height=0.32\linewidth,",
        r"xlabel={Quantile policy},",
        r"ylabel={Model},",
        "xmin=-0.5, xmax=" + _tex_float(len(QUANTILE_ORDER) - 0.5, 1) + ",",
        "ymin=-0.5, ymax=" + _tex_float(len(MODEL_ORDER) - 0.5, 1) + ",",
        "xtick={" + ",".join(str(i) for i in range(len(QUANTILE_ORDER))) + "},",
        "xticklabels={" + ",".join(_latex_escape(q) for q in QUANTILE_ORDER) + "},",
        "ytick={" + ",".join(str(i) for i in range(len(MODEL_ORDER))) + "},",
        "yticklabels={" + ",".join(_latex_escape(r) for r in MODEL_ORDER) + "},",
        r"y dir=reverse,",
        r"point meta min=0.00,",
        r"point meta max=1.00,",
        r"colormap={rq2blue}{rgb255(0cm)=(228,241,247); rgb255(1cm)=(197,225,239); rgb255(2cm)=(158,201,226); rgb255(3cm)=(108,176,214); rgb255(4cm)=(60,147,194); rgb255(5cm)=(34,110,156); rgb255(6cm)=(13,74,112)},",
        r"colorbar,",
        r"colorbar style={ylabel={Relative Value of Forecast}, ytick={0,0.25,0.5,0.75,1}, yticklabels={0\%,25\%,50\%,75\%,100\%}, tick label style={font=\small}},",
        r"]",
        r"\addplot[scatter, only marks, mark=none, draw=none, opacity=0, scatter/use mapped color={draw opacity=0, fill opacity=0}] coordinates {",
        r"(0,0) [0.000]",
        r"(0,0) [1.000]",
        r"};",
        *cells,
        r"\end{axis}",
        r"\end{tikzpicture}",
        rf"\caption{{{CAPTION}}}",
        r"\label{fig:relative_value_of_forecast_heatmap}",
        r"\end{figure}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_outputs(rq2_root: Path) -> dict[str, Path]:
    data, warnings = build_relative_vof(rq2_root)
    csv_path = rq2_root / "result_section" / "csv" / "relative_value_of_forecast_heatmap_data.csv"
    png_path = rq2_root / "result_section" / "figures" / "relative_value_of_forecast_heatmap.png"
    pdf_path = rq2_root / "result_section" / "figures" / "relative_value_of_forecast_heatmap.pdf"
    tex_path = rq2_root / "result_section" / "latex_figures" / "relative_value_of_forecast_heatmap.tex"
    write_csv(data, csv_path)
    plot_heatmap(data, png_path, pdf_path)
    write_latex(data, tex_path)
    for warning in warnings:
        print(f"[WARN] {warning}")
    return {"csv": csv_path, "png": png_path, "pdf": pdf_path, "tex": tex_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rq2-root", type=Path, default=DEFAULT_RQ2_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = build_outputs(args.rq2_root)
    for key, path in outputs.items():
        print(f"[OK] {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
