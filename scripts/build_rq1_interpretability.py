#!/usr/bin/env python3
"""Build lightweight interpretability add-on outputs for the RQ1 ML benchmark.

This script reuses existing model artifacts. It does not train models, run HPO,
run simulations, or compute full SHAP. XGB SHAP values are consumed only from
existing importance reports. TFT interpretation is copied as visual artifacts
and explicitly not labelled as SHAP.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from energy_trading.evaluation.style import THESIS_PALETTE, apply_geo_style, thesis_titlecase


TARGET_LABELS = {
    "target_da_price": "DA price",
    "target_afrr_capacity_price_pos": "aFRR capacity price pos",
    "target_afrr_capacity_price_neg": "aFRR capacity price neg",
    "target_afrr_activation_price_vwap_pos": "aFRR activation price pos",
    "target_afrr_activation_price_vwap_neg": "aFRR activation price neg",
    "target_afrr_activation_rate_pos": "aFRR activation rate pos",
    "target_afrr_activation_rate_neg": "aFRR activation rate neg",
}

TARGET_STEMS = {
    "target_da_price": "da",
    "target_afrr_capacity_price_pos": "afrr_afrr_capacity_price_pos",
    "target_afrr_capacity_price_neg": "afrr_afrr_capacity_price_neg",
    "target_afrr_activation_price_vwap_pos": "afrr_afrr_activation_price_vwap_pos",
    "target_afrr_activation_price_vwap_neg": "afrr_afrr_activation_price_vwap_neg",
    "target_afrr_activation_rate_pos": "afrr_afrr_activation_rate_pos",
    "target_afrr_activation_rate_neg": "afrr_afrr_activation_rate_neg",
}

TFT_PREFIX = {
    "target_da_price": "da_target_da_price",
    "target_afrr_capacity_price_pos": "afrr_target_afrr_capacity_price_pos",
    "target_afrr_capacity_price_neg": "afrr_target_afrr_capacity_price_neg",
    "target_afrr_activation_price_vwap_pos": "afrr_target_afrr_activation_price_vwap_pos",
    "target_afrr_activation_price_vwap_neg": "afrr_target_afrr_activation_price_vwap_neg",
    "target_afrr_activation_rate_pos": "afrr_target_afrr_activation_rate_pos",
    "target_afrr_activation_rate_neg": "afrr_target_afrr_activation_rate_neg",
}

MODEL_LABELS = {"xgb": "XGB", "linear": "RLQR", "tft": "TFT"}


@dataclass(frozen=True)
class WarningRow:
    target: str
    model: str
    warning_type: str
    message: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_latest_run_dir(model: str, model_runs_root: Path) -> Path:
    latest = model_runs_root / f"latest_{model}.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    run_id = payload.get("run_id")
    if not run_id:
        manifest_path = payload.get("manifest_path")
        if manifest_path:
            run_id = Path(str(manifest_path)).parts[0]
    if not run_id:
        raise ValueError(f"Could not resolve run_id from {latest}")
    return model_runs_root / str(run_id)


def feature_group(feature: str) -> str:
    """Map a raw feature name to a thesis-readable feature group."""
    f = str(feature).lower()
    if any(x in f for x in ["afrr_activation_rate", "is_activated", "activation_rate"]):
        return "activation history"
    if any(x in f for x in ["afrr_capacity", "capacity_price", "capacity_offered"]):
        return "capacity market history"
    if any(x in f for x in ["afrr_marginal", "activation_price", "imbalance", "nrv", "system_stress"]):
        return "aFRR system state"
    if "price" in f or "spread" in f:
        return "price history"
    if "residual" in f:
        return "residual load"
    if "load" in f:
        return "load"
    if any(x in f for x in ["solar", "wind", "renewable"]):
        return "renewable generation"
    if any(x in f for x in ["hour", "day", "week", "month", "holiday", "season", "morning", "evening", "night", "afternoon", "calendar", "payday"]):
        return "calendar/time"
    if any(x in f for x in ["lag", "rolling", "mean_", "std_", "ewma", "diff", "ramp", "tminus"]):
        return "lag/rolling statistics"
    if any(x in f for x in ["cross", "border", "fr_", "nl_", "be_", "at_", "ch_", "pl_", "cz_", "neighbor"]):
        return "cross-border / neighboring markets"
    if any(x in f for x in ["temp", "weather", "wind_speed", "irradiance"]):
        return "weather"
    return "other"


def _target_label(target: str) -> str:
    return TARGET_LABELS.get(target, target.replace("target_", "").replace("_", " "))


def _latex_escape(value: Any) -> str:
    s = str(value)
    for old, new in {
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
    }.items():
        s = s.replace(old, new)
    return s


def _fmt(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return str(value)
    return f"{x:.4g}" if np.isfinite(x) else "-"


def _write_latex_table(path: Path, headers: list[str], rows: list[list[Any]], caption: str, label: str) -> Path:
    align = "l" * len(headers)
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        f"    \\caption{{{_latex_escape(caption)}}}",
        f"    \\label{{{label}}}",
        rf"    \begin{{tabular}}{{@{{}}{align}@{{}}}}",
        r"        \toprule",
        "        " + " & ".join(r"\textbf{" + _latex_escape(h) + "}" for h in headers) + r" \\",
        r"        \midrule",
    ]
    for row in rows:
        vals = [_fmt(v) if isinstance(v, (float, int, np.floating, np.integer)) else _latex_escape(v) for v in row]
        lines.append("        " + " & ".join(vals) + r" \\")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _latex_color_name(role: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", role).lower()


def _latex_color_defs() -> list[str]:
    return [
        f"\\definecolor{{{_latex_color_name(role)}}}{{HTML}}{{{hex_color.lstrip('#').upper()}}}"
        for role, hex_color in THESIS_PALETTE.items()
    ]


def _tex_num(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(x):
        return "nan"
    return f"{x:.6g}"


def _tex_symbol(value: Any) -> str:
    raw = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    if not raw:
        raw = "x"
    if raw[0].isdigit():
        raw = "x_" + raw
    return raw


def _write_native_bar_figure(
    path: Path,
    *,
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    caption: str,
    label: str,
    xlabel: str,
    color: str = "secondary",
    placement: str = "htbp",
) -> Path | None:
    if df.empty:
        return None
    d = df.copy()
    d = d.sort_values(value_col, ascending=True)
    labels = d[label_col].astype(str).tolist()
    symbols = [_tex_symbol(x) for x in labels]
    coords = " ".join(f"({_tex_num(v)},{_tex_symbol(lbl)})" for lbl, v in zip(d[label_col], d[value_col]))
    lines = [
        r"% Requires: \usepackage{pgfplots}",
        r"% Requires: \usepackage{xcolor}",
        r"% Recommended in preamble: \pgfplotsset{compat=1.18}",
        *_latex_color_defs(),
        rf"\begin{{figure}}[{placement}]",
        r"    \centering",
        r"    \resizebox{\linewidth}{!}{%",
        r"        \begin{tikzpicture}",
        r"            \begin{axis}[",
        r"                xbar,",
        r"                bar width=12pt,",
        r"                width=0.98\textwidth,",
        r"                height=8cm,",
        rf"                xlabel={{{_latex_escape(xlabel)}}},",
        "                symbolic y coords={" + ",".join(symbols) + "},",
        "                ytick={" + ",".join(symbols) + "},",
        "                yticklabels={" + ",".join(_latex_escape(x) for x in labels) + "},",
        r"                yticklabel style={font=\scriptsize},",
        r"                axis lines*=left,",
        r"                xmin=0,",
        r"            ]",
        rf"                \addplot[xbar, fill={color}, draw={color}] coordinates {{{coords}}};",
        r"            \end{axis}",
        r"        \end{tikzpicture}}",
        f"    \\caption{{{_latex_escape(thesis_titlecase(caption))}}}",
        f"    \\label{{{label}}}",
        r"\end{figure}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_group_heatmap_figure(path: Path, *, group_df: pd.DataFrame, caption: str, label: str) -> Path | None:
    if group_df.empty:
        return None
    top = group_df[group_df["group_rank"].le(5)].copy()
    if top.empty:
        return None
    top["panel"] = top["target"].astype(str) + " | " + top["model"].astype(str)
    x_labels = top["panel"].drop_duplicates().tolist()
    y_labels = sorted(top["feature_group"].drop_duplicates().tolist())
    lines = [
        r"% Requires: \usepackage{pgfplots}",
        r"% Requires: \usepackage{xcolor}",
        r"% Recommended in preamble: \pgfplotsset{compat=1.18}",
        *_latex_color_defs(),
        r"\begin{figure}[htbp]",
        r"    \centering",
        r"    \resizebox{\linewidth}{!}{%",
        r"        \begin{tikzpicture}",
        r"            \begin{axis}[",
        r"                width=0.98\textwidth,",
        r"                height=8cm,",
        r"                view={0}{90},",
        r"                colorbar,",
        r"                colormap/Blues,",
        r"                x tick label style={rotate=55, anchor=east, font=\scriptsize},",
        "                symbolic x coords={" + ",".join(_tex_symbol(x) for x in x_labels) + "},",
        "                xtick={" + ",".join(_tex_symbol(x) for x in x_labels) + "},",
        "                xticklabels={" + ",".join(_latex_escape(x) for x in x_labels) + "},",
        "                symbolic y coords={" + ",".join(_tex_symbol(y) for y in y_labels) + "},",
        "                ytick={" + ",".join(_tex_symbol(y) for y in y_labels) + "},",
        "                yticklabels={" + ",".join(_latex_escape(y) for y in y_labels) + "},",
        r"            ]",
        r"                \addplot[matrix plot*, point meta=explicit] coordinates {",
    ]
    for _, row in top.iterrows():
        lines.append(f"                    ({_tex_symbol(row['panel'])},{_tex_symbol(row['feature_group'])}) [{_tex_num(row['importance_share'])}]")
    lines.extend(
        [
            r"                };",
            r"            \end{axis}",
            r"        \end{tikzpicture}}",
            f"    \\caption{{{_latex_escape(thesis_titlecase(caption))}}}",
            f"    \\label{{{label}}}",
            r"\end{figure}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def deterministic_top_n(df: pd.DataFrame, group_cols: list[str], value_col: str, n: int) -> pd.DataFrame:
    sort_cols = group_cols + [value_col, "feature"]
    asc = [True] * len(group_cols) + [False, True]
    ranked = df.sort_values(sort_cols, ascending=asc).copy()
    ranked["rank"] = ranked.groupby(group_cols, sort=False).cumcount() + 1
    return ranked[ranked["rank"].le(int(n))].reset_index(drop=True)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _inventory_row(target: str, model: str, artifact_type: str, path: Path, exists: bool, used: bool, reason: str = "") -> dict[str, Any]:
    return {
        "target": _target_label(target),
        "model": MODEL_LABELS.get(model, model),
        "artifact_type": artifact_type,
        "path": str(path),
        "exists": bool(exists),
        "used": bool(used),
        "reason_if_not_used": reason,
    }


def _collect_xgb(run_dir: Path, section_root: Path, inventory: list[dict[str, Any]], warnings_out: list[WarningRow]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reports = run_dir / "reports"
    for target, stem in TARGET_STEMS.items():
        report = reports / f"xgboost_{stem}_importance_report.csv"
        exists = report.exists()
        inventory.append(_inventory_row(target, "xgb", "xgb_importance_csv", report, exists, exists))
        if not exists:
            warnings_out.append(WarningRow(_target_label(target), "XGB", "xgb_importance_missing", f"Missing XGB importance report: {report}"))
            continue
        df = pd.read_csv(report)
        if "feature" not in df.columns:
            warnings_out.append(WarningRow(_target_label(target), "XGB", "feature_names_missing", f"No feature column in {report}"))
            continue
        if "shap_mean_abs" in df.columns:
            for _, r in df.iterrows():
                value = float(r.get("shap_mean_abs", np.nan))
                if np.isfinite(value):
                    rows.append(
                        {
                            "target": _target_label(target),
                            "target_key": target,
                            "model": "XGB",
                            "feature": str(r["feature"]),
                            "feature_group": feature_group(str(r["feature"])),
                            "importance_type": "existing_xgb_mean_abs_shap",
                            "importance_value": value,
                            "caveat": "Existing SHAP summary reused; no new SHAP computation in this script.",
                        }
                    )
        else:
            warnings_out.append(WarningRow(_target_label(target), "XGB", "shap_unavailable", f"No shap_mean_abs column in {report}"))
        if "xgboost_gain" in df.columns:
            for _, r in df.iterrows():
                value = float(r.get("xgboost_gain", np.nan))
                if np.isfinite(value):
                    rows.append(
                        {
                            "target": _target_label(target),
                            "target_key": target,
                            "model": "XGB",
                            "feature": str(r["feature"]),
                            "feature_group": feature_group(str(r["feature"])),
                            "importance_type": "native_xgb_gain",
                            "importance_value": value,
                            "caveat": "Native XGB gain, not directly comparable to SHAP, TFT attention, or RLQR coefficients.",
                        }
                    )

        shap_png = reports / f"xgboost_{stem}_shap_summary.png"
        shap_dst = section_root / "appendix" / "figures" / f"xgb_shap_summary_{target.removeprefix('target_')}.png"
        used = _copy_if_exists(shap_png, shap_dst)
        inventory.append(_inventory_row(target, "xgb", "xgb_shap_summary_png", shap_png, shap_png.exists(), used, "" if used else "SHAP plot unavailable."))
        if not used:
            warnings_out.append(WarningRow(_target_label(target), "XGB", "shap_unavailable", f"Missing existing SHAP plot: {shap_png}"))

        imp_png = reports / f"xgboost_{stem}_top20_feature_importance.png"
        imp_dst = section_root / "appendix" / "figures" / f"xgb_feature_importance_{target.removeprefix('target_')}.png"
        used = _copy_if_exists(imp_png, imp_dst)
        inventory.append(_inventory_row(target, "xgb", "xgb_feature_importance_png", imp_png, imp_png.exists(), used, "" if used else "Native XGB feature-importance plot unavailable."))
    return pd.DataFrame(rows)


def _collect_tft(run_dir: Path, section_root: Path, inventory: list[dict[str, Any]], warnings_out: list[WarningRow]) -> pd.DataFrame:
    reports = run_dir / "reports"
    for target, prefix in TFT_PREFIX.items():
        copied_any = False
        for kind, suffix in [
            ("tft_feature_relevance_png", "feature_importance"),
            ("tft_attention_png", "attention_interpretation"),
            ("tft_attention_history_png", "attention_history_tminus"),
        ]:
            src = reports / f"{prefix}_{suffix}.png"
            dst = section_root / "appendix" / "figures" / f"tft_attention_or_feature_importance_{target.removeprefix('target_')}_{suffix}.png"
            used = _copy_if_exists(src, dst)
            copied_any = copied_any or used
            inventory.append(_inventory_row(target, "tft", kind, src, src.exists(), used, "" if used else "TFT visual interpretation artifact unavailable."))
        if copied_any:
            warnings_out.append(WarningRow(_target_label(target), "TFT", "tft_not_shap", "TFT attention/feature-relevance plots are used as visual diagnostics only; they are not SHAP values."))
        else:
            warnings_out.append(WarningRow(_target_label(target), "TFT", "tft_importance_missing", "No TFT attention or feature-relevance artifact found."))
    return pd.DataFrame(
        columns=["target", "target_key", "model", "feature", "feature_group", "importance_type", "importance_value", "caveat"]
    )


def _patch_torch_load_cpu():
    import torch

    original = torch.load

    def cpu_load(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device("cpu"))
        return original(*args, **kwargs)

    torch.load = cpu_load
    return original


def _linear_model_path(run_dir: Path, target: str) -> Path:
    if target == "target_da_price":
        return run_dir / "models" / "linear_da_da_price_model.joblib"
    return run_dir / "models" / f"linear_afrr_{target.removeprefix('target_')}_model.joblib"


def _collect_linear(run_dir: Path, section_root: Path, inventory: list[dict[str, Any]], warnings_out: list[WarningRow], *, lead: int, quantile: str) -> pd.DataFrame:
    import joblib
    import torch

    rows: list[dict[str, Any]] = []
    original_load = _patch_torch_load_cpu()
    try:
        for target in TARGET_STEMS:
            model_path = _linear_model_path(run_dir, target)
            exists = model_path.exists()
            inventory.append(_inventory_row(target, "linear", "rlqr_model_joblib", model_path, exists, exists))
            if not exists:
                warnings_out.append(WarningRow(_target_label(target), "RLQR", "rlqr_model_missing", f"Missing RLQR model artifact: {model_path}"))
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    payload = joblib.load(model_path)
                lead_models = payload.get("lead_models", {})
                q_payload = lead_models.get(int(lead), {}).get(str(quantile))
                if q_payload is None:
                    warnings_out.append(WarningRow(_target_label(target), "RLQR", "rlqr_lead_quantile_missing", f"No {quantile} coefficients for lead {lead} in {model_path}"))
                    continue
                model = q_payload.get("model")
                imputer = q_payload.get("imputer")
                scaler = q_payload.get("scaler")
                if model is None or not hasattr(model, "get_coefficients"):
                    warnings_out.append(WarningRow(_target_label(target), "RLQR", "rlqr_coefficients_missing", f"Model exposes no get_coefficients(): {model_path}"))
                    continue
                coef_obj = model.get_coefficients()
                weights = coef_obj[0] if isinstance(coef_obj, tuple) else coef_obj
                weights = np.asarray(weights, dtype=float)
                q_idx = int(q_payload.get("quantile_index", 0))
                if weights.ndim == 2:
                    coefs = weights[q_idx, :]
                else:
                    coefs = weights.ravel()
                if imputer is None or not hasattr(imputer, "feature_names_in_"):
                    warnings_out.append(WarningRow(_target_label(target), "RLQR", "feature_names_missing", f"No feature_names_in_ on imputer for {model_path}"))
                    continue
                features = list(map(str, imputer.feature_names_in_))
                if len(features) != len(coefs):
                    warnings_out.append(WarningRow(_target_label(target), "RLQR", "feature_names_mismatch", f"Feature/coefficient length mismatch in {model_path}"))
                    continue
                has_scaling = scaler is not None and hasattr(scaler, "scale_")
                if not has_scaling:
                    warnings_out.append(WarningRow(_target_label(target), "RLQR", "raw_coefficients_used", "Feature scaling metadata unavailable; raw coefficients are not scale comparable."))
                else:
                    warnings_out.append(WarningRow(_target_label(target), "RLQR", "robust_scaled_coefficients", "RLQR coefficients are on robust-scaled feature inputs and are not comparable to SHAP magnitudes."))
                for feature, coef in zip(features, coefs):
                    rows.append(
                        {
                            "target": _target_label(target),
                            "target_key": target,
                            "model": "RLQR",
                            "feature": feature,
                            "feature_group": feature_group(feature),
                            "importance_type": "robust_scaled_p50_coefficient_abs" if has_scaling else "raw_p50_coefficient_abs",
                            "importance_value": abs(float(coef)),
                            "caveat": (
                                f"{quantile} lead-{lead} coefficient magnitude on robust-scaled inputs; not comparable to XGB SHAP or TFT attention."
                                if has_scaling
                                else f"{quantile} lead-{lead} raw coefficient magnitude; feature scaling unavailable."
                            ),
                        }
                    )
            except Exception as exc:
                warnings_out.append(WarningRow(_target_label(target), "RLQR", "rlqr_extraction_failed", f"Could not extract RLQR coefficients from {model_path}: {exc}"))
    finally:
        torch.load = original_load

    coef_df = pd.DataFrame(rows)
    if not coef_df.empty:
        top = deterministic_top_n(coef_df, ["target", "model", "importance_type"], "importance_value", 20)
        for target, g in top.groupby("target_key", sort=True):
            fig_path = section_root / "appendix" / "figures" / f"rlqr_coefficients_{target.removeprefix('target_')}.png"
            _plot_feature_bars(g, fig_path, f"RLQR p50 lead-{lead} coefficients: {_target_label(target)}", "Coefficient magnitude")
            _write_native_bar_figure(
                section_root / "appendix" / "latex_figures" / f"rlqr_coefficients_{target.removeprefix('target_')}.tex",
                df=g.sort_values("importance_value", ascending=False).head(20),
                label_col="feature",
                value_col="importance_value",
                caption=f"Top RLQR coefficient magnitudes for {_target_label(target)} at lead {lead} and {quantile}.",
                label=f"fig:rq1-interpretability-rlqr-coefficients-{target.removeprefix('target_').replace('_', '-')}",
                xlabel="Coefficient magnitude",
                placement="p",
            )
    return coef_df


def _plot_feature_bars(df: pd.DataFrame, path: Path, title: str, xlabel: str) -> None:
    import matplotlib.pyplot as plt

    d = df.sort_values("importance_value", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(d["feature"], d["importance_value"], color=THESIS_PALETTE["secondary"], edgecolor=THESIS_PALETTE["neutral_dark"])
    ax.set_title(thesis_titlecase(title))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _feature_group_table(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=["target", "target_key", "model", "feature_group", "importance_type", "importance_value", "importance_share", "group_rank"])
    primary = long_df[
        long_df["importance_type"].isin(["existing_xgb_mean_abs_shap", "robust_scaled_p50_coefficient_abs", "raw_p50_coefficient_abs"])
    ].copy()
    if primary.empty:
        return pd.DataFrame()
    grouped = (
        primary.groupby(["target", "target_key", "model", "feature_group", "importance_type"], as_index=False)["importance_value"]
        .sum()
        .sort_values(["target", "model", "importance_value"], ascending=[True, True, False])
    )
    total = grouped.groupby(["target", "model", "importance_type"])["importance_value"].transform("sum")
    grouped["importance_share"] = np.where(total > 0, grouped["importance_value"] / total, np.nan)
    grouped["group_rank"] = grouped.groupby(["target", "model", "importance_type"])["importance_value"].rank(method="first", ascending=False).astype(int)
    return grouped


def _overlap_table(top_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary = top_features[
        top_features["importance_type"].isin(["existing_xgb_mean_abs_shap", "robust_scaled_p50_coefficient_abs", "raw_p50_coefficient_abs"])
    ]
    for target, g in primary.groupby("target", sort=True):
        by_model = {m: set(df["feature"].astype(str)) for m, df in g.groupby("model")}
        xgb = by_model.get("XGB", set())
        rlqr = by_model.get("RLQR", set())
        inter = xgb & rlqr
        union = xgb | rlqr
        rows.append(
            {
                "target": target,
                "models_compared": "XGB vs RLQR",
                "top_feature_overlap_count": len(inter),
                "top_feature_union_count": len(union),
                "jaccard_overlap": len(inter) / len(union) if union else np.nan,
                "overlapping_features": "; ".join(sorted(inter)[:10]),
            }
        )
    return pd.DataFrame(rows)


def _plot_group_importance(group_df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    top = group_df[group_df["group_rank"].le(5)].copy()
    if top.empty:
        return
    top["panel"] = top["target"] + " | " + top["model"]
    panels = top["panel"].drop_duplicates().tolist()
    groups = sorted(top["feature_group"].drop_duplicates().tolist())
    matrix = pd.DataFrame(0.0, index=groups, columns=panels)
    for _, row in top.iterrows():
        matrix.loc[row["feature_group"], row["panel"]] = float(row["importance_share"])

    fig, ax = plt.subplots(figsize=(max(11, 0.42 * len(panels)), max(5.5, 0.35 * len(groups))))
    im = ax.imshow(matrix.values, aspect="auto", cmap="Blues", vmin=0.0)
    ax.set_xticks(np.arange(len(panels)))
    ax.set_xticklabels(panels, rotation=55, ha="right")
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_title(thesis_titlecase("Top feature-group relevance by target and model"))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Within-model importance share")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _summary_rows(top_features: pd.DataFrame, overlap: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    primary = top_features[
        top_features["importance_type"].isin(["existing_xgb_mean_abs_shap", "robust_scaled_p50_coefficient_abs", "raw_p50_coefficient_abs"])
    ]
    for target in [_target_label(t) for t in TARGET_STEMS]:
        g = primary[primary["target"].eq(target)]
        xgb = ", ".join(g[g["model"].eq("XGB")].sort_values("rank")["feature_group"].drop_duplicates().head(3).tolist()) or "Unavailable"
        rlqr = ", ".join(g[g["model"].eq("RLQR")].sort_values("rank")["feature_group"].drop_duplicates().head(3).tolist()) or "Unavailable"
        ov = overlap[overlap["target"].eq(target)]
        agreement = "XGB/RLQR top-feature overlap unavailable"
        if not ov.empty and np.isfinite(float(ov["jaccard_overlap"].iloc[0])):
            agreement = f"XGB/RLQR Jaccard {float(ov['jaccard_overlap'].iloc[0]):.2f}"
        rows.append(
            [
                target,
                "Visual artifact only; see appendix",
                xgb,
                rlqr,
                agreement,
                "Importance values are method-specific and not directly comparable.",
            ]
        )
    return rows


def build_interpretability_outputs(
    *,
    model_runs_root: Path,
    out_dir: Path,
    lead: int = 24,
    quantile: str = "p50",
) -> dict[str, Any]:
    apply_geo_style()
    for rel in [
        "result_section/figures",
        "result_section/latex_figures",
        "result_section/tables",
        "appendix/figures",
        "appendix/latex_figures",
        "appendix/tables",
        "backup/csv",
        "backup/diagnostics",
        "backup/warnings",
    ]:
        (out_dir / rel).mkdir(parents=True, exist_ok=True)
    for tex in out_dir.rglob("*.tex"):
        if "latex_figures" in tex.parts and r"\includegraphics" in tex.read_text(encoding="utf-8", errors="ignore"):
            tex.unlink()

    xgb_dir = _load_latest_run_dir("xgboost", model_runs_root)
    tft_dir = _load_latest_run_dir("tft", model_runs_root)
    linear_dir = _load_latest_run_dir("linear", model_runs_root)

    inventory: list[dict[str, Any]] = []
    warning_rows: list[WarningRow] = [
        WarningRow("All targets", "All models", "cross_model_not_directly_comparable", "SHAP values, XGB gain, TFT attention, and RLQR coefficients are different objects; compare ranks/groups only as descriptive diagnostics.")
    ]
    frames = [
        _collect_xgb(xgb_dir, out_dir, inventory, warning_rows),
        _collect_tft(tft_dir, out_dir, inventory, warning_rows),
        _collect_linear(linear_dir, out_dir, inventory, warning_rows, lead=lead, quantile=quantile),
    ]
    long_df = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if not long_df.empty:
        long_df = long_df.sort_values(["target", "model", "importance_type", "importance_value", "feature"], ascending=[True, True, True, False, True])
    top_features = deterministic_top_n(long_df, ["target", "model", "importance_type"], "importance_value", 10) if not long_df.empty else pd.DataFrame()
    group_df = _feature_group_table(long_df)
    overlap = _overlap_table(top_features) if not top_features.empty else pd.DataFrame()

    long_path = out_dir / "backup" / "csv" / "feature_importance_long.csv"
    top_path = out_dir / "backup" / "csv" / "feature_importance_top_features.csv"
    group_path = out_dir / "backup" / "csv" / "feature_group_importance.csv"
    overlap_path = out_dir / "backup" / "csv" / "feature_overlap_by_target.csv"
    long_df.to_csv(long_path, index=False)
    top_features.to_csv(top_path, index=False)
    group_df.to_csv(group_path, index=False)
    overlap.to_csv(overlap_path, index=False)

    inventory_path = out_dir / "backup" / "diagnostics" / "interpretability_artifact_inventory.csv"
    pd.DataFrame(inventory).to_csv(inventory_path, index=False)
    warnings_path = out_dir / "backup" / "warnings" / "interpretability_warnings.csv"
    pd.DataFrame([w.__dict__ for w in warning_rows]).to_csv(warnings_path, index=False)

    fig_path = out_dir / "result_section" / "figures" / "rq1_interpretability_top_feature_groups.png"
    _plot_group_importance(group_df, fig_path)
    _write_group_heatmap_figure(
        out_dir / "result_section" / "latex_figures" / "rq1_interpretability_top_feature_groups.tex",
        group_df=group_df,
        caption="Top feature groups by target and model for the RQ1 interpretability add-on. Shares are normalized within each model and target; magnitudes are not comparable across model classes.",
        label="fig:rq1-interpretability-top-feature-groups",
    )

    summary_tex = _write_latex_table(
        out_dir / "result_section" / "tables" / "rq1_interpretability_summary.tex",
        ["Target", "TFT main drivers", "XGB main drivers", "RLQR main drivers", "Cross-model agreement", "Caveat"],
        _summary_rows(top_features, overlap),
        "Compact interpretability summary for the RQ1 ML benchmark.",
        "tab:rq1_interpretability_summary",
    )

    appendix_tex = _write_latex_table(
        out_dir / "appendix" / "tables" / "feature_importance_top_features_test.tex",
        ["Target", "Model", "Rank", "Feature", "Feature group", "Importance type", "Importance value", "Caveat"],
        top_features[["target", "model", "rank", "feature", "feature_group", "importance_type", "importance_value", "caveat"]].head(180).values.tolist()
        if not top_features.empty
        else [],
        "Top feature-level interpretability diagnostics for the RQ1 ML benchmark.",
        "tab:rq1_interpretability_top_features_test",
    )

    manifest = {
        "description": "Lightweight interpretability add-on for RQ1 ML benchmark.",
        "methods": {
            "XGB": "Existing mean absolute SHAP and native gain importance reports are reused; no new SHAP computation.",
            "TFT": "Existing attention/feature-relevance images are routed to appendix; not SHAP.",
            "RLQR": f"{quantile} lead-{lead} robust-scaled coefficient magnitudes are extracted from saved artifacts where available.",
        },
        "model_run_dirs": {"xgb": str(xgb_dir), "tft": str(tft_dir), "linear": str(linear_dir)},
        "outputs": {
            "result_section": [str(fig_path), str(summary_tex)],
            "appendix": [str(appendix_tex)],
            "backup": [str(long_path), str(top_path), str(group_path), str(overlap_path), str(inventory_path), str(warnings_path)],
        },
    }
    manifest_path = out_dir / "backup" / "diagnostics" / "interpretability_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build lightweight RQ1 interpretability outputs.")
    p.add_argument("--model-runs-root", default="artifacts/model_runs")
    p.add_argument("--out-dir", default="artifacts/benchmark/rq1_ml_model_benchmark/diagnostics/rq1_interpretability")
    p.add_argument("--lead", type=int, default=24)
    p.add_argument("--quantile", default="p50")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_interpretability_outputs(
        model_runs_root=Path(args.model_runs_root),
        out_dir=Path(args.out_dir),
        lead=int(args.lead),
        quantile=str(args.quantile),
    )
    print(f"[OK] RQ1 interpretability outputs written: {Path(args.out_dir)}")
    print(f"[OK] result_section={len(manifest['outputs']['result_section'])} appendix={len(manifest['outputs']['appendix'])} backup={len(manifest['outputs']['backup'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
