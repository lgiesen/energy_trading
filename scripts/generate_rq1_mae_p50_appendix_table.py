#!/usr/bin/env python3
"""Generate an RQ1 appendix table for absolute MAE p50 by target and model."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DETAILED_CSV = Path(
    "artifacts/benchmark/rq1_ml_model_benchmark/_raw_outputs/4_1_1_full_unweighted/csv/"
    "rq1_4_1_1_forecast_metrics_full_detailed_test.csv"
)
DEFAULT_LONG_CSV = Path(
    "artifacts/benchmark/rq1_ml_model_benchmark/_raw_outputs/4_1_1_full_unweighted/csv/"
    "rq1_4_1_1_forecast_metrics_full_long.csv"
)
DEFAULT_DETAILED_TEX = Path(
    "artifacts/benchmark/rq1_ml_model_benchmark/_raw_outputs/4_1_1_full_unweighted/latex/"
    "rq1_4_1_1_forecast_metrics_full_detailed_test.tex"
)
DEFAULT_OUT_ROOT = Path("artifacts/benchmark/rq1_ml_model_benchmark")

MODEL_ORDER = ["RLQR", "XGB", "TFT"]
TARGET_ORDER = [
    "pred_da_price",
    "pred_afrr_capacity_price_pos",
    "pred_afrr_capacity_price_neg",
    "pred_afrr_activation_price_pos",
    "pred_afrr_activation_price_neg",
    "pred_afrr_activation_rate_pos",
    "pred_afrr_activation_rate_neg",
]
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
MODEL_ALIASES = {
    "linear": "RLQR",
    "rlqr": "RLQR",
    "xgb": "XGB",
    "xgboost": "XGB",
    "tft": "TFT",
}
CAPTION = (
    "Absolute MAE p50 by forecast target and model. Values report the mean absolute error of the median forecast "
    "in the original target units."
)
NOTE = (
    "Values are not normalized and therefore should be interpreted within each forecast target because target units "
    "and scales differ."
)


def _safe_float(value: Any) -> float:
    x = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(x) if pd.notna(x) and math.isfinite(float(x)) else math.nan


def _normalize_label(value: Any) -> str:
    text = str(value).strip()
    text = text.replace("−", "-").replace("–", "-")
    text = text.replace(r"\textminus", "-").replace(r"$-$", "-")
    text = re.sub(r"\\[A-Za-z]+\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\", "")
    return " ".join(text.split())


def _canonical_target(value: Any) -> str | None:
    raw = str(value).strip()
    normalized = _normalize_label(raw)
    keys = [
        raw,
        normalized,
        normalized.lower(),
        normalized.lower().replace("_", " "),
    ]
    if raw.startswith(("pred_", "target_")):
        keys.append(raw)
    for key in keys:
        if key in TARGET_ALIASES:
            return TARGET_ALIASES[key]
    return None


def _canonical_model(value: Any) -> str | None:
    raw = str(value).strip()
    if raw in MODEL_ORDER:
        return raw
    return MODEL_ALIASES.get(raw.lower())


def _latex_escape(value: Any) -> str:
    text = str(value)
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
        text = text.replace(old, new)
    return text


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(col).lower().replace(" ", "_"): str(col) for col in df.columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None


def _sort_targets(table: pd.DataFrame) -> pd.DataFrame:
    order = {target: idx for idx, target in enumerate(TARGET_ORDER)}
    out = table.copy()
    out["_target_order"] = out["target"].map(order)
    out = out.loc[out["_target_order"].notna()].sort_values("_target_order")
    return out.drop(columns=["_target_order"]).reset_index(drop=True)


def _read_detailed_wide_csv(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path)
    required = {"target", "metric"}
    missing = sorted(required - set(df.columns))
    model_cols = [model for model in MODEL_ORDER if model in df.columns]
    if missing or not model_cols:
        raise ValueError(
            f"{path} is not a detailed-wide metric CSV. Missing={missing}, detected model columns={model_cols}."
        )
    rows: list[dict[str, Any]] = []
    matches = df.loc[df["metric"].astype(str).str.lower().eq("mae_p50") | df["metric"].astype(str).str.lower().eq("mae p50")]
    if matches.empty:
        raise ValueError(f"No MAE p50 metric rows found in {path}.")
    for _, row in matches.iterrows():
        target = _canonical_target(row["target"])
        if target is None:
            continue
        rows.append(
            {
                "target": target,
                "target_label": TARGET_LABELS[target],
                **{model: _safe_float(row[model]) if model in row else math.nan for model in MODEL_ORDER},
            }
        )
    table = _sort_targets(pd.DataFrame(rows))
    if table.empty:
        raise ValueError(f"No recognized MAE p50 target rows found in {path}.")
    detected = {"input_format": "detailed_wide_csv", "target_column": "target", "metric_column": "metric", "model_columns": model_cols}
    return table, detected


def _read_long_csv(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path)
    target_col = _find_col(df, ["target", "forecast_target", "target_label", "pred_col"])
    model_col = _find_col(df, ["model", "model_label"])
    mae_col = _find_col(df, ["mae_p50"])
    missing = [
        name
        for name, col in [("target/forecast target label", target_col), ("model", model_col), ("mae_p50", mae_col)]
        if col is None
    ]
    if missing:
        raise ValueError(f"{path} is missing required long-format column(s): {', '.join(missing)}.")
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        target = _canonical_target(row[target_col])
        model = _canonical_model(row[model_col])
        if target is None or model is None:
            continue
        rows.append({"target": target, "target_label": TARGET_LABELS[target], "model": model, "mae_p50": _safe_float(row[mae_col])})
    long = pd.DataFrame(rows)
    if long.empty:
        raise ValueError(f"No recognized target/model MAE p50 rows found in {path}.")
    table = (
        long.pivot_table(index=["target", "target_label"], columns="model", values="mae_p50", aggfunc="mean")
        .reset_index()
    )
    for model in MODEL_ORDER:
        if model not in table.columns:
            table[model] = math.nan
    table = _sort_targets(table[["target", "target_label", *MODEL_ORDER]])
    detected = {"input_format": "long_csv", "target_column": target_col, "model_column": model_col, "mae_column": mae_col}
    return table, detected


def _strip_tex_cell(cell: str) -> str:
    text = cell.strip().rstrip("\\").strip()
    text = text.replace(r"\%", "%").replace(r"\&", "&").replace(r"\_", "_")
    text = text.replace(r"$-$", "-")
    text = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\shortstack\{.*?\}", "", text)
    return " ".join(text.split())


def _read_detailed_tex(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if "&" not in stripped or not stripped.endswith(r"\\"):
            continue
        cells = [_strip_tex_cell(part) for part in stripped.split("&")]
        if len(cells) < 5:
            continue
        target = _canonical_target(cells[0])
        metric = _normalize_label(cells[1]).lower()
        if target is None or metric != "mae p50":
            continue
        rows.append(
            {
                "target": target,
                "target_label": TARGET_LABELS[target],
                "RLQR": _safe_float(cells[2]),
                "XGB": _safe_float(cells[3]),
                "TFT": _safe_float(cells[4]),
            }
        )
    table = _sort_targets(pd.DataFrame(rows))
    if table.empty:
        raise ValueError(f"No MAE p50 rows could be parsed from {path}.")
    detected = {
        "input_format": "detailed_latex_table",
        "target_column": "Target",
        "metric_column": "Metric",
        "model_columns": MODEL_ORDER,
    }
    return table, detected


def read_mae_table(inputs: list[Path], fallback_tex: Path | None) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    errors: list[str] = []
    for path in inputs:
        if not path.exists():
            errors.append(f"{path}: missing")
            continue
        try:
            if path.name.endswith("_long.csv"):
                table, detected = _read_long_csv(path)
            else:
                table, detected = _read_detailed_wide_csv(path)
            return table, path, detected
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    if fallback_tex is not None and fallback_tex.exists():
        table, detected = _read_detailed_tex(fallback_tex)
        return table, fallback_tex, detected
    joined = "\n  - ".join(errors) if errors else "no candidate CSV inputs supplied"
    raise FileNotFoundError(
        "Could not read absolute MAE p50 data from the requested RQ1 metric outputs.\n"
        f"Attempted:\n  - {joined}\n"
        f"Fallback LaTeX table: {fallback_tex if fallback_tex is not None else 'disabled'}"
    )


def _format_value(target: str, value: Any) -> str:
    x = _safe_float(value)
    if not math.isfinite(x):
        return "--"
    if "activation_rate" in str(target):
        return f"{x:.4f}"
    return f"{x:.2f}"


def write_outputs(table: pd.DataFrame, out_root: Path) -> tuple[Path, Path]:
    csv_path = out_root / "appendix" / "csv" / "mae_p50_absolute_by_target_model.csv"
    tex_path = out_root / "appendix" / "tables" / "mae_p50_absolute_by_target_model.tex"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    table[["target", "target_label", *MODEL_ORDER]].to_csv(csv_path, index=False)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{CAPTION}}}",
        r"\label{tab:appendix_mae_p50_absolute_by_target_model}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"\textbf{Forecast target} & \textbf{RLQR} & \textbf{XGB} & \textbf{TFT} \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        cells = [
            TARGET_LABELS_TEX.get(str(row["target"]), _latex_escape(row["target_label"])),
            *[_format_value(str(row["target"]), row[model]) for model in MODEL_ORDER],
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.35em}",
            rf"\parbox{{0.88\linewidth}}{{\footnotesize {_latex_escape(NOTE)}}}",
            r"\end{table}",
            "",
        ]
    )
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, tex_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detailed-csv", type=Path, default=DEFAULT_DETAILED_CSV)
    parser.add_argument("--long-csv", type=Path, default=DEFAULT_LONG_CSV)
    parser.add_argument("--fallback-tex", type=Path, default=DEFAULT_DETAILED_TEX)
    parser.add_argument("--no-fallback-tex", action="store_true", help="Disable parsing the existing detailed LaTeX metric table.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fallback_tex = None if args.no_fallback_tex else args.fallback_tex
    table, input_path, detected = read_mae_table([args.detailed_csv, args.long_csv], fallback_tex)
    csv_path, tex_path = write_outputs(table, args.out_root)

    print(f"[OK] input file used: {input_path}")
    print(f"[OK] detected columns: {detected}")
    print(f"[OK] detected models: {', '.join(MODEL_ORDER)}")
    print(f"[OK] detected targets: {', '.join(table['target_label'].astype(str).tolist())}")
    print(f"[OK] output CSV: {csv_path}")
    print(f"[OK] output LaTeX table: {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
