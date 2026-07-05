#!/usr/bin/env python3
"""Generate RQ1 relative MAE p50 heatmap and raw MBE p50 table."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_trading.evaluation.style import GEO_SEQUENTIAL_BLUE, THESIS_PALETTE, apply_geo_style


DEFAULT_INPUT = Path(
    "artifacts/benchmark/rq1_ml_model_benchmark/_raw_outputs/4_1_1_full_unweighted/csv/"
    "rq1_4_1_1_forecast_metrics_full_detailed_test.csv"
)
DEFAULT_OUT_ROOT = Path("artifacts/benchmark/rq1_ml_model_benchmark")

MODEL_ORDER = ["RLQR", "XGB", "TFT"]
RELATIVE_MODEL_ORDER = MODEL_ORDER
TABLE_MODEL_ORDER = MODEL_ORDER
TARGET_ORDER = [
    "pred_da_price",
    "pred_afrr_capacity_price_pos",
    "pred_afrr_capacity_price_neg",
    "pred_afrr_activation_price_pos",
    "pred_afrr_activation_price_neg",
    "pred_afrr_activation_rate_pos",
    "pred_afrr_activation_rate_neg",
]
TARGET_ALIASES = {
    "da price": "pred_da_price",
    "pred_da_price": "pred_da_price",
    "target_da_price": "pred_da_price",
    "afrr capacity price +": "pred_afrr_capacity_price_pos",
    "afrr capacity price positive": "pred_afrr_capacity_price_pos",
    "pred_afrr_capacity_price_pos": "pred_afrr_capacity_price_pos",
    "target_afrr_capacity_price_pos": "pred_afrr_capacity_price_pos",
    "afrr capacity price -": "pred_afrr_capacity_price_neg",
    "afrr capacity price $-$": "pred_afrr_capacity_price_neg",
    "afrr capacity price negative": "pred_afrr_capacity_price_neg",
    "pred_afrr_capacity_price_neg": "pred_afrr_capacity_price_neg",
    "target_afrr_capacity_price_neg": "pred_afrr_capacity_price_neg",
    "afrr activation price +": "pred_afrr_activation_price_pos",
    "afrr activation price positive": "pred_afrr_activation_price_pos",
    "pred_afrr_activation_price_pos": "pred_afrr_activation_price_pos",
    "target_afrr_activation_price_pos": "pred_afrr_activation_price_pos",
    "afrr activation price -": "pred_afrr_activation_price_neg",
    "afrr activation price $-$": "pred_afrr_activation_price_neg",
    "afrr activation price negative": "pred_afrr_activation_price_neg",
    "pred_afrr_activation_price_neg": "pred_afrr_activation_price_neg",
    "target_afrr_activation_price_neg": "pred_afrr_activation_price_neg",
    "afrr activation rate +": "pred_afrr_activation_rate_pos",
    "afrr activation rate positive": "pred_afrr_activation_rate_pos",
    "pred_afrr_activation_rate_pos": "pred_afrr_activation_rate_pos",
    "target_afrr_activation_rate_pos": "pred_afrr_activation_rate_pos",
    "afrr activation rate -": "pred_afrr_activation_rate_neg",
    "afrr activation rate $-$": "pred_afrr_activation_rate_neg",
    "afrr activation rate negative": "pred_afrr_activation_rate_neg",
    "pred_afrr_activation_rate_neg": "pred_afrr_activation_rate_neg",
    "target_afrr_activation_rate_neg": "pred_afrr_activation_rate_neg",
}
TARGET_LABELS = {
    "pred_da_price": "DA price",
    "pred_afrr_capacity_price_pos": "aFRR capacity price +",
    "pred_afrr_capacity_price_neg": "aFRR capacity price -",
    "pred_afrr_activation_price_pos": "aFRR activation price +",
    "pred_afrr_activation_price_neg": "aFRR activation price -",
    "pred_afrr_activation_rate_pos": "aFRR activation rate +",
    "pred_afrr_activation_rate_neg": "aFRR activation rate -",
}
TARGET_LABELS_TEX = {
    "pred_da_price": "DA price",
    "pred_afrr_capacity_price_pos": "aFRR capacity price +",
    "pred_afrr_capacity_price_neg": r"aFRR capacity price $-$",
    "pred_afrr_activation_price_pos": "aFRR activation price +",
    "pred_afrr_activation_price_neg": r"aFRR activation price $-$",
    "pred_afrr_activation_rate_pos": "aFRR activation rate +",
    "pred_afrr_activation_rate_neg": r"aFRR activation rate $-$",
}
MODEL_ALIASES = {
    "rlqr": "RLQR",
    "linear": "RLQR",
    "xgb": "XGB",
    "xgboost": "XGB",
    "tft": "TFT",
}
RELATIVE_MAE_CAPTION = (
    "Relative MAE $p50$ compares median forecast error across models and forecast targets using RLQR as the reference "
    "benchmark. Cell values below 1 indicate lower MAE than RLQR for the same target, while values above 1 indicate higher MAE."
)
MBE_TABLE_CAPTION = (
    "MBE p50 compares the median-forecast bias across forecast targets and models. Positive values indicate "
    "overprediction, while negative values indicate underprediction."
)
LEGACY_STEMS = [
    "mae_p50_by_target_model",
    "bias_p50_by_target_model",
    "mae_bias_p50_by_target_model",
    "mae_p50_price_targets_by_model",
    "mae_p50_activation_rate_targets_by_model",
    "bias_p50_price_targets_by_model",
    "bias_p50_activation_rate_targets_by_model",
]


def _safe_float(value: Any) -> float:
    x = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(x) if pd.notna(x) and math.isfinite(float(x)) else math.nan


def _format_cell(value: float, *, decimals: int = 2) -> str:
    if not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{decimals}f}"


def _format_mbe(value: float) -> str:
    if not math.isfinite(float(value)):
        return "--"
    abs_value = abs(float(value))
    if abs_value < 0.01 and abs_value > 0:
        return f"{float(value):.4f}"
    return f"{float(value):.2f}"


def _tex_float(value: float, digits: int = 4) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def _tex_color_def(name: str, hex_color: str) -> str:
    hex_color = str(hex_color).lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return rf"\definecolor{{{name}}}{{rgb}}{{{r:.4f},{g:.4f},{b:.4f}}}"


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


def _canonical_target(value: Any) -> str | None:
    raw = str(value).strip()
    key = raw.lower().replace("−", "-").replace(r"\textminus", "-").replace("–", "-")
    key = " ".join(key.replace("_", " ").split()) if not raw.startswith(("pred_", "target_")) else raw
    if raw in TARGET_ALIASES:
        return TARGET_ALIASES[raw]
    return TARGET_ALIASES.get(key)


def _canonical_model(value: Any) -> str | None:
    raw = str(value).strip()
    if raw in TABLE_MODEL_ORDER:
        return raw
    return MODEL_ALIASES.get(raw.lower())


def _ordered_targets(df: pd.DataFrame) -> pd.DataFrame:
    order = {target: idx for idx, target in enumerate(TARGET_ORDER)}
    out = df.copy()
    out["_target_order"] = out["target"].map(order)
    out = out.loc[out["_target_order"].notna()].sort_values("_target_order")
    return out.drop(columns=["_target_order"]).reset_index(drop=True)


def _read_metric_table(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing RQ1 detailed forecast metric CSV: {path}. "
            "Regenerate 4.1.1 without --skip-raw-generation first."
        )
    source = pd.read_csv(path)
    available_metrics: list[str] = []
    if {"target", "metric"}.issubset(source.columns) and any(model in source.columns for model in TABLE_MODEL_ORDER):
        available_metrics = sorted(source["metric"].astype(str).unique().tolist())
        mae = _extract_wide_metric(source, ["mae_p50"], output_metric="mae_p50")
        mbe = _extract_wide_metric(source, ["bias_p50", "p50_bias"], output_metric="mbe_p50")
        detected = {
            "input_format": "detailed_wide",
            "target_column": "target",
            "metric_column": "metric",
            "model_columns": [m for m in TABLE_MODEL_ORDER if m in source.columns],
            "mae_metric": "mae_p50",
            "mbe_metric": "bias_p50" if "bias_p50" in available_metrics else "p50_bias",
        }
        return mae, mbe, available_metrics, detected
    mae, mbe, available_metrics, detected = _extract_long_metrics(source)
    return mae, mbe, available_metrics, detected


def _extract_wide_metric(df: pd.DataFrame, metric_names: list[str], *, output_metric: str) -> pd.DataFrame:
    matches = df.loc[df["metric"].astype(str).isin(metric_names)].copy()
    if matches.empty:
        raise ValueError(f"Missing required metric row. Expected one of: {metric_names}")
    rows: list[dict[str, Any]] = []
    for _, row in matches.iterrows():
        target = _canonical_target(row["target"])
        if target is None:
            continue
        out = {
            "target": target,
            "target_label": TARGET_LABELS[target],
            "metric": output_metric,
        }
        for model in TABLE_MODEL_ORDER:
            out[model] = _safe_float(row[model]) if model in row else math.nan
        rows.append(out)
    result = _ordered_targets(pd.DataFrame(rows))
    if result.empty:
        raise ValueError(f"No recognized target rows available for metric {output_metric}.")
    return result


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(col).lower().replace(" ", "_"): str(col) for col in df.columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None


def _extract_long_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    target_col = _find_col(df, ["target", "forecast_target", "target_label", "pred_col"])
    model_col = _find_col(df, ["model_label", "model"])
    mae_col = _find_col(df, ["mae_p50"])
    mbe_col = _find_col(df, ["bias_p50", "p50_bias", "mbe_p50"])
    missing = [
        name
        for name, col in {
            "target": target_col,
            "model": model_col,
            "mae_p50": mae_col,
            "bias_p50 or p50_bias": mbe_col,
        }.items()
        if col is None
    ]
    if missing:
        raise ValueError(
            "Input metric CSV is neither detailed-wide nor recognized long format. "
            f"Missing columns: {missing}. Available columns: {list(df.columns)}"
        )
    assert target_col is not None and model_col is not None and mae_col is not None and mbe_col is not None
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        target = _canonical_target(row[target_col])
        model = _canonical_model(row[model_col])
        if target is None or model is None:
            continue
        rows.append(
            {
                "target": target,
                "target_label": TARGET_LABELS[target],
                "model": model,
                "mae_p50": _safe_float(row[mae_col]),
                "mbe_p50": _safe_float(row[mbe_col]),
            }
        )
    long = pd.DataFrame(rows)
    if long.empty:
        raise ValueError("No recognized target/model rows were found in the long metric CSV.")

    def pivot(metric: str) -> pd.DataFrame:
        p = long.pivot_table(index=["target", "target_label"], columns="model", values=metric, aggfunc="first").reset_index()
        for model in TABLE_MODEL_ORDER:
            if model not in p.columns:
                p[model] = math.nan
        p["metric"] = metric
        return _ordered_targets(p[["target", "target_label", "metric", *TABLE_MODEL_ORDER]])

    detected = {
        "input_format": "long",
        "target_column": target_col,
        "model_column": model_col,
        "mae_column": mae_col,
        "mbe_column": mbe_col,
    }
    return pivot("mae_p50"), pivot("mbe_p50"), [mae_col, mbe_col], detected


def _build_relative_mae(mae: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for _, row in mae.iterrows():
        target = str(row["target"])
        rlqr = _safe_float(row["RLQR"])
        if not math.isfinite(rlqr) or abs(rlqr) <= 1e-12:
            skipped.append({"target": target, "reason": "missing_or_zero_rlqr_mae_p50"})
            continue
        for model in RELATIVE_MODEL_ORDER:
            mae_value = _safe_float(row[model])
            rows.append(
                {
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "model": model,
                    "mae_p50": mae_value,
                    "rlqr_mae_p50": rlqr,
                    "relative_mae_p50": mae_value / rlqr if math.isfinite(mae_value) else math.nan,
                }
            )
    if not rows:
        raise ValueError("No relative MAE p50 rows could be computed because RLQR MAE p50 was missing or zero.")
    long = pd.DataFrame(rows)
    wide = long.pivot(index=["target", "target_label"], columns="model", values="relative_mae_p50").reset_index()
    for model in RELATIVE_MODEL_ORDER:
        if model not in wide.columns:
            wide[model] = math.nan
    wide = _ordered_targets(wide[["target", "target_label", *RELATIVE_MODEL_ORDER]])
    return long, pd.DataFrame(skipped)


def _relative_cmap() -> ListedColormap:
    return ListedColormap(
        [GEO_SEQUENTIAL_BLUE[f"seq_{i}"] for i in range(1, 8)],
        "rq1_relative_mae",
    )


def _relative_color_indices(values: np.ndarray) -> np.ndarray:
    indices = np.full(values.shape, math.nan, dtype=float)
    for yi in range(values.shape[0]):
        row = values[yi, :]
        finite_mask = np.isfinite(row)
        finite = row[finite_mask]
        if not finite.size:
            continue
        row_min = float(np.min(finite))
        row_max = float(np.max(finite))
        if abs(row_max - row_min) <= 1e-12:
            indices[yi, finite_mask] = 3.0
            continue
        scaled = (row[finite_mask] - row_min) / (row_max - row_min)
        indices[yi, finite_mask] = np.clip(np.rint(scaled * 6.0), 0, 6)
    return indices


def _plot_relative_mae_heatmap(data: pd.DataFrame, *, png_path: Path, pdf_path: Path, skip_png: bool, skip_pdf: bool) -> list[Path]:
    outputs: list[Path] = []
    if skip_png and skip_pdf:
        return outputs
    apply_geo_style()
    values = data[RELATIVE_MODEL_ORDER].to_numpy(dtype=float)
    color_indices = _relative_color_indices(values)
    cmap = _relative_cmap()
    cmap.set_bad("#F2F2F2")
    norm = BoundaryNorm(np.arange(-0.5, 7.5, 1.0), cmap.N)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    im = ax.imshow(np.ma.masked_invalid(color_indices), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(RELATIVE_MODEL_ORDER)), labels=RELATIVE_MODEL_ORDER)
    ax.set_yticks(np.arange(len(data)), labels=data["target_label"].tolist())
    ax.set_xlabel("Model")
    ax.set_ylabel("")
    ax.set_title("MAE p50 Relative to RLQR")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.025, ticks=[0, 3, 6])
    cbar.set_label("Within-target relative MAE")
    cbar.ax.set_yticklabels(["lower", "middle", "higher"])
    for yi in range(values.shape[0]):
        for xi in range(values.shape[1]):
            value = values[yi, xi]
            if not math.isfinite(float(value)):
                color = THESIS_PALETTE["neutral_dark"]
            else:
                color_index = color_indices[yi, xi]
                color = "white" if math.isfinite(float(color_index)) and color_index >= 5 else THESIS_PALETTE["neutral_dark"]
            ax.text(xi, yi, _format_cell(value), ha="center", va="center", fontsize=9.5, color=color)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if not skip_png:
        fig.savefig(png_path, dpi=220)
        outputs.append(png_path)
    if not skip_pdf:
        fig.savefig(pdf_path)
        outputs.append(pdf_path)
    plt.close(fig)
    return outputs


def _cell_fill(color_index: float) -> tuple[str, int]:
    if not math.isfinite(float(color_index)):
        return "rqOneMissing", 0
    seq_index = int(np.clip(round(float(color_index)), 0, 6)) + 1
    return f"rqOneSeq{seq_index}", seq_index


def _latex_target_label(target: str) -> str:
    return TARGET_LABELS_TEX.get(str(target), _latex_escape(target))


def _write_relative_mae_latex(data: pd.DataFrame, *, path: Path) -> Path:
    values = data[RELATIVE_MODEL_ORDER].to_numpy(dtype=float)
    color_indices = _relative_color_indices(values)
    cells: list[str] = []
    for yi, row in data.reset_index(drop=True).iterrows():
        for xi, model in enumerate(RELATIVE_MODEL_ORDER):
            value = _safe_float(row[model])
            fill, color_index = _cell_fill(color_indices[yi, xi])
            text_color = "white" if color_index >= 6 else "rqOneNeutral"
            cells.extend(
                [
                    rf"\filldraw[fill={fill}, draw=white, line width=0.6pt] "
                    rf"(axis cs:{xi - 0.5},{yi - 0.5}) rectangle (axis cs:{xi + 0.5},{yi + 0.5});",
                    rf"\node[text={text_color}, font=\small] at (axis cs:{xi},{yi}) {{{_format_cell(value)}}};",
                ]
            )
    lines = [
        r"% Requires \usepackage{pgfplots}",
        r"% Requires \pgfplotsset{compat=1.18}",
        r"\begin{figure}[htbp]",
        r"\centering",
        _tex_color_def("rqOneNeutral", THESIS_PALETTE["neutral_dark"]),
        *[_tex_color_def(f"rqOneSeq{i}", GEO_SEQUENTIAL_BLUE[f"seq_{i}"]) for i in range(1, 8)],
        _tex_color_def("rqOneMissing", "#F2F2F2"),
        _tex_color_def("rqOneGrid", "#D8D8D8"),
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"tick align=outside,",
        r"axis line style={rqOneNeutral},",
        r"tick style={rqOneNeutral},",
        r"label style={font=\small},",
        r"tick label style={font=\small},",
        r"title style={font=\normalfont\small},",
        r"grid=major,",
        r"grid style={rqOneGrid!55, line width=0.2pt},",
        r"title={MAE p50 Relative to RLQR},",
        r"width=0.60\linewidth,",
        r"height=0.46\linewidth,",
        r"xlabel={Model},",
        r"ylabel={},",
        "xmin=-0.5, xmax=" + _tex_float(len(RELATIVE_MODEL_ORDER) - 0.5, 1) + ",",
        "ymin=-0.5, ymax=" + _tex_float(len(data) - 0.5, 1) + ",",
        "xtick={" + ",".join(str(i) for i in range(len(RELATIVE_MODEL_ORDER))) + "},",
        "xticklabels={" + ",".join(RELATIVE_MODEL_ORDER) + "},",
        "ytick={" + ",".join(str(i) for i in range(len(data))) + "},",
        "yticklabels={" + ",".join(_latex_target_label(t) for t in data["target"]) + "},",
        r"yticklabel style={align=right},",
        r"y dir=reverse,",
        "point meta min=0,",
        "point meta max=6,",
        "colormap={rq1relmae}{"
        + " ".join(
            f"rgb255({i - 1}cm)=({int(GEO_SEQUENTIAL_BLUE[f'seq_{i}'][1:3], 16)},"
            f"{int(GEO_SEQUENTIAL_BLUE[f'seq_{i}'][3:5], 16)},"
            f"{int(GEO_SEQUENTIAL_BLUE[f'seq_{i}'][5:7], 16)});"
            for i in range(1, 8)
        )
        + "},",
        r"colorbar,",
        r"colorbar style={ylabel={Within-target relative MAE}, ylabel style={font=\small}, ytick={0,3,6}, yticklabels={lower,middle,higher}, tick label style={font=\small}},",
        r"]",
        r"\addplot[scatter, only marks, mark=none, draw=none, opacity=0, scatter/use mapped color={draw opacity=0, fill opacity=0}] coordinates {",
        "(0,0) [0]",
        "(0,0) [6]",
        r"};",
        *cells,
        r"\end{axis}",
        r"\end{tikzpicture}",
        rf"\caption{{{RELATIVE_MAE_CAPTION}}}",
        r"\label{fig:mae_p50_relative_to_rlqr_by_target_model}",
        r"\end{figure}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_mbe_table(mbe: pd.DataFrame, *, path: Path) -> Path:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{MBE_TABLE_CAPTION}}}",
        r"\label{tab:mbe_p50_raw_by_target_model}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Forecast target} & \textbf{RLQR} & \textbf{XGB} & \textbf{TFT} \\",
        r"\midrule",
    ]
    for _, row in mbe.iterrows():
        vals = " & ".join(_format_mbe(_safe_float(row[model])) for model in TABLE_MODEL_ORDER)
        lines.append(rf"{_latex_target_label(str(row['target']))} & {vals} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _remove_legacy_outputs(out_root: Path) -> list[Path]:
    removed: list[Path] = []
    subdirs = {
        "result_section/csv": [".csv"],
        "result_section/figures": [".png", ".pdf", ".svg"],
        "result_section/latex_figures": [".tex"],
        "result_section/latex_tables": [".tex"],
    }
    for subdir, suffixes in subdirs.items():
        for stem in LEGACY_STEMS:
            for suffix in suffixes:
                path = out_root / subdir / f"{stem}{suffix}"
                if path.exists():
                    path.unlink()
                    removed.append(path)
    return removed


def build_outputs(
    input_path: Path,
    out_root: Path,
    *,
    skip_csv: bool = False,
    skip_png: bool = False,
    skip_pdf: bool = False,
) -> dict[str, Any]:
    removed = _remove_legacy_outputs(out_root)
    mae, mbe, available_metrics, detected_columns = _read_metric_table(input_path)
    summary = build_outputs_from_metric_frames(
        mae=mae,
        mbe=mbe,
        out_root=out_root,
        skip_csv=skip_csv,
        skip_png=skip_png,
        skip_pdf=skip_pdf,
    )
    summary["input"] = input_path
    summary["available_metrics"] = available_metrics
    summary["detected_columns"] = detected_columns
    summary["removed_legacy_outputs"] = removed + summary["removed_legacy_outputs"]
    return summary


def build_outputs_from_metric_frames(
    *,
    mae: pd.DataFrame,
    mbe: pd.DataFrame,
    out_root: Path,
    skip_csv: bool = False,
    skip_png: bool = False,
    skip_pdf: bool = False,
) -> dict[str, Any]:
    removed = _remove_legacy_outputs(out_root)
    relative_long, skipped_targets = _build_relative_mae(mae)
    relative_wide = relative_long.pivot(index=["target", "target_label"], columns="model", values="relative_mae_p50").reset_index()
    for model in RELATIVE_MODEL_ORDER:
        if model not in relative_wide.columns:
            relative_wide[model] = math.nan
    relative_wide = _ordered_targets(relative_wide[["target", "target_label", *RELATIVE_MODEL_ORDER]])

    csv_dir = out_root / "result_section" / "csv"
    fig_dir = out_root / "result_section" / "figures"
    latex_fig_dir = out_root / "result_section" / "latex_figures"
    latex_table_dir = out_root / "result_section" / "latex_tables"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    latex_fig_dir.mkdir(parents=True, exist_ok=True)
    latex_table_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    relative_csv = csv_dir / "mae_p50_relative_to_rlqr_by_target_model.csv"
    mbe_csv = csv_dir / "mbe_p50_raw_by_target_model.csv"
    if not skip_csv:
        relative_long.to_csv(relative_csv, index=False)
        outputs.append(relative_csv)
        mbe[["target", "target_label", *TABLE_MODEL_ORDER]].to_csv(mbe_csv, index=False)
        outputs.append(mbe_csv)

    outputs.extend(
        _plot_relative_mae_heatmap(
            relative_wide,
            png_path=fig_dir / "mae_p50_relative_to_rlqr_by_target_model.png",
            pdf_path=fig_dir / "mae_p50_relative_to_rlqr_by_target_model.pdf",
            skip_png=skip_png,
            skip_pdf=skip_pdf,
        )
    )
    outputs.append(_write_relative_mae_latex(relative_wide, path=latex_fig_dir / "mae_p50_relative_to_rlqr_by_target_model.tex"))
    outputs.append(_write_mbe_table(mbe, path=latex_table_dir / "mbe_p50_raw_by_target_model.tex"))

    return {
        "input": None,
        "available_metrics": ["mae_p50", "mbe_p50"],
        "detected_columns": {"input_format": "in_memory"},
        "outputs": outputs,
        "removed_legacy_outputs": removed,
        "models_relative": RELATIVE_MODEL_ORDER,
        "models_table": TABLE_MODEL_ORDER,
        "targets": relative_wide["target_label"].tolist(),
        "skipped_targets": skipped_targets.to_dict("records") if not skipped_targets.empty else [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Detailed RQ1 forecast metric CSV.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="RQ1 benchmark output root.")
    parser.add_argument("--skip-csv", action="store_true", help="Do not write plotted CSV exports.")
    parser.add_argument("--skip-png", action="store_true", help="Do not write PNG figure exports.")
    parser.add_argument("--skip-pdf", action="store_true", help="Do not write PDF figure exports.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_outputs(args.input, args.out_root, skip_csv=args.skip_csv, skip_png=args.skip_png, skip_pdf=args.skip_pdf)
    print(f"[OK] input: {summary['input']}")
    print(f"[OK] available metrics: {', '.join(summary['available_metrics'])}")
    print(f"[OK] detected columns: {summary['detected_columns']}")
    print(f"[OK] relative heatmap models: {', '.join(summary['models_relative'])}")
    print(f"[OK] MBE table models: {', '.join(summary['models_table'])}")
    print(f"[OK] targets: {', '.join(summary['targets'])}")
    if summary["skipped_targets"]:
        print(f"[WARN] skipped targets for relative MAE: {summary['skipped_targets']}")
    for path in summary["removed_legacy_outputs"]:
        print(f"[OK] removed legacy point-error output: {path}")
    for path in summary["outputs"]:
        print(f"[OK] generated: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
