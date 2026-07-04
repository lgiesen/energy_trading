#!/usr/bin/env python3
"""Diagnose validation-test shift for negative aFRR capacity prices.

The script reads existing prepared targets and saved model prediction exports only.
It does not train models, rerun forecasts, or modify model artifacts.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats
    from scipy.spatial import distance as scipy_distance
except Exception as exc:  # pragma: no cover - scipy is expected in the thesis env
    raise RuntimeError("scipy is required for validation-test shift diagnostics") from exc


TARGET_COLUMN = "target_afrr_capacity_price_neg"
CANONICAL_TARGET = "pred_afrr_capacity_price_neg"
TARGET_LABEL = "Negative aFRR capacity price"
QUANTILE_COLUMNS = ["p10", "p30", "p50", "p70", "p90"]
QUANTILE_LEVELS = {"p10": 0.10, "p30": 0.30, "p50": 0.50, "p70": 0.70, "p90": 0.90}
MODEL_ORDER = ["RLQR", "XGB", "TFT"]
MODEL_COLORS = {"RLQR": "#4C4C4C", "XGB": "#226E9C", "TFT": "#8A1C7C"}

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "artifacts/benchmark/rq1_ml_model_benchmark/appendix/validation_test_shift_capacity_price_neg"
THESIS_ROOT = Path(
    "/Users/leori/Desktop/ uni/3 Master IS/25 MA/"
    "MA___Elevated_Energy_Trading__Forecasting_Expected_Profit_From_Participating_in_the_Day_Ahead_and_Balancing_Markets"
)
THESIS_REL_ROOT = Path("figures/4-results/rq1_ml_model_benchmark/appendix/validation_test_shift_capacity_price_neg")


@dataclass(frozen=True)
class PredictionSpec:
    model: str
    split: str
    path: Path


PREDICTION_SPECS = [
    PredictionSpec(
        "RLQR",
        "validation",
        REPO_ROOT
        / "artifacts/model_runs/linear_20260531_092342/predictions/linear_afrr_afrr_capacity_price_neg_val_pred_afrr_capacity_price_neg_long.parquet",
    ),
    PredictionSpec(
        "RLQR",
        "test",
        REPO_ROOT
        / "artifacts/model_runs/linear_20260531_092342/predictions/linear_afrr_afrr_capacity_price_neg_test_pred_afrr_capacity_price_neg_long.parquet",
    ),
    PredictionSpec(
        "XGB",
        "validation",
        REPO_ROOT
        / "artifacts/model_runs/xgb_20260530_151122/predictions/xgboost_afrr_afrr_capacity_price_neg_val_pred_afrr_capacity_price_neg_long.parquet",
    ),
    PredictionSpec(
        "XGB",
        "test",
        REPO_ROOT
        / "artifacts/model_runs/xgb_20260530_151122/predictions/xgboost_afrr_afrr_capacity_price_neg_test_pred_afrr_capacity_price_neg_long.parquet",
    ),
    PredictionSpec(
        "TFT",
        "validation",
        REPO_ROOT
        / "artifacts/model_runs/tft_20260530_150841/predictions/afrr_target_afrr_capacity_price_neg_pred_afrr_capacity_price_neg_long.parquet",
    ),
    PredictionSpec(
        "TFT",
        "test",
        REPO_ROOT
        / "artifacts/model_runs/tft_20260530_150841/predictions/afrr_target_afrr_capacity_price_neg_pred_afrr_capacity_price_neg_long_test.parquet",
    ),
]


def _ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "csv": root / "csv",
        "tables": root / "tables",
        "figures": root / "figures",
        "latex_figures": root / "latex_figures",
        "diagnostics": root / "diagnostics",
        "text": root / "text",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _latex_escape(value: object) -> str:
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
    return "".join(replacements.get(ch, ch) for ch in text)


def _format_float(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return str(value)


def _write_latex_table(
    df: pd.DataFrame,
    path: Path,
    *,
    caption: str,
    label: str,
    columns: list[str],
    digits: int = 3,
) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \small",
        f"    \\caption{{{_latex_escape(caption)}}}",
        f"    \\label{{{label}}}",
        r"    \begin{tabular}{" + "l" * len(columns) + r"}",
        r"        \toprule",
        "        " + " & ".join(rf"\textbf{{{_latex_escape(col)}}}" for col in columns) + r" \\",
        r"        \midrule",
    ]
    for _, row in df[columns].iterrows():
        cells: list[str] = []
        for col in columns:
            value = row[col]
            if isinstance(value, (int, np.integer)):
                cells.append(f"{int(value):,}")
            elif isinstance(value, (float, np.floating)):
                cells.append(_format_float(value, digits))
            else:
                cells.append(_latex_escape(value))
        lines.append("        " + " & ".join(cells) + r" \\")
    lines.extend([r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_latex_figure(path: Path, *, image_name: str, caption: str, label: str, width: str = r"\linewidth") -> None:
    image_rel = THESIS_REL_ROOT / "figures" / image_name
    lines = [
        r"\begin{figure}[htbp]",
        r"    \centering",
        rf"    \includegraphics[width={width}]{{{image_rel.as_posix()}}}",
        f"    \\caption{{{_latex_escape(caption)}}}",
        f"    \\label{{{label}}}",
        r"\end{figure}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_target_split(split: str) -> pd.DataFrame:
    file_name = "val.parquet" if split == "validation" else "test.parquet"
    path = REPO_ROOT / "data/model_input/afrr" / file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared target file: {path}")
    df = pd.read_parquet(path)
    required = {"timestamp_utc", TARGET_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    out = df[["timestamp_utc", TARGET_COLUMN]].copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    out = out.rename(columns={"timestamp_utc": "target_time_utc", TARGET_COLUMN: "y_true"})
    out["split"] = split
    return out.dropna(subset=["target_time_utc", "y_true"])


def _normalize_prediction_columns(df: pd.DataFrame, spec: PredictionSpec) -> pd.DataFrame:
    required = {"target_time_utc", "snapshot_time_utc", "lead_time_h", *QUANTILE_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{spec.path} is missing required columns: {sorted(missing)}")
    out = df[["snapshot_time_utc", "target_time_utc", "lead_time_h", *QUANTILE_COLUMNS]].copy()
    out["snapshot_time_utc"] = pd.to_datetime(out["snapshot_time_utc"], utc=True)
    out["target_time_utc"] = pd.to_datetime(out["target_time_utc"], utc=True)
    out["lead_time_h"] = pd.to_numeric(out["lead_time_h"], errors="coerce")
    for col in QUANTILE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["model"] = spec.model
    out["split"] = spec.split
    return out


def _load_predictions(targets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    log_rows: list[dict[str, object]] = []
    for spec in PREDICTION_SPECS:
        if not spec.path.exists():
            raise FileNotFoundError(f"Missing prediction file for {spec.model} {spec.split}: {spec.path}")
        raw = pd.read_parquet(spec.path)
        pred = _normalize_prediction_columns(raw, spec)
        n_pred = len(pred)
        joined = pred.merge(targets[spec.split][["target_time_utc", "y_true"]], on="target_time_utc", how="left")
        missing_truth = int(joined["y_true"].isna().sum())
        joined = joined.dropna(subset=["y_true", *QUANTILE_COLUMNS]).copy()
        frames.append(joined)
        log_rows.append(
            {
                "model": spec.model,
                "split": spec.split,
                "prediction_file": str(spec.path.relative_to(REPO_ROOT)),
                "target_file": f"data/model_input/afrr/{'val' if spec.split == 'validation' else 'test'}.parquet",
                "target_column": TARGET_COLUMN,
                "canonical_prediction_target": CANONICAL_TARGET,
                "timestamp_join": "target_time_utc == timestamp_utc",
                "prediction_rows": n_pred,
                "missing_truth_after_join": missing_truth,
                "rows_used": len(joined),
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(log_rows)


def _distribution_summary(targets: pd.DataFrame, validation_values: pd.Series) -> pd.DataFrame:
    v95 = validation_values.quantile(0.95)
    v99 = validation_values.quantile(0.99)
    v05 = validation_values.quantile(0.05)
    v01 = validation_values.quantile(0.01)
    rows: list[dict[str, object]] = []
    for split, group in targets.groupby("split", sort=False):
        s = group["y_true"].dropna()
        rows.append(
            {
                "split": split,
                "n": len(s),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(ddof=1),
                "min": s.min(),
                "max": s.max(),
                "p01": s.quantile(0.01),
                "p05": s.quantile(0.05),
                "p10": s.quantile(0.10),
                "p25": s.quantile(0.25),
                "p75": s.quantile(0.75),
                "p90": s.quantile(0.90),
                "p95": s.quantile(0.95),
                "p99": s.quantile(0.99),
                "iqr": s.quantile(0.75) - s.quantile(0.25),
                "skewness": s.skew(),
                "kurtosis": s.kurtosis(),
                "near_zero_share": float((s.abs() <= 1e-9).mean()),
                "negative_share": float((s < 0).mean()),
                "share_above_validation_p95": float((s > v95).mean()),
                "share_above_validation_p99": float((s > v99).mean()),
                "share_below_validation_p05": float((s < v05).mean()),
                "share_below_validation_p01": float((s < v01).mean()),
            }
        )
    return pd.DataFrame(rows)


def _shift_tests(validation: pd.Series, test: pd.Series) -> pd.DataFrame:
    validation = validation.dropna()
    test = test.dropna()
    ks = stats.ks_2samp(validation, test, alternative="two-sided", mode="auto")
    mw = stats.mannwhitneyu(validation, test, alternative="two-sided")
    bf = stats.levene(validation, test, center="median")
    wasserstein = stats.wasserstein_distance(validation, test)

    pooled_min = float(min(validation.min(), test.min()))
    pooled_max = float(max(validation.max(), test.max()))
    if pooled_min == pooled_max:
        js_distance = 0.0
    else:
        bins = np.linspace(pooled_min, pooled_max, 61)
        hist_v, _ = np.histogram(validation, bins=bins, density=False)
        hist_t, _ = np.histogram(test, bins=bins, density=False)
        p = hist_v.astype(float) + 1e-12
        q = hist_t.astype(float) + 1e-12
        p /= p.sum()
        q /= q.sum()
        js_distance = float(scipy_distance.jensenshannon(p, q))
    rows = [
        {
            "test": "Kolmogorov-Smirnov",
            "statistic": ks.statistic,
            "p_value": ks.pvalue,
            "interpretation": "distribution differs" if ks.pvalue < 0.05 else "no clear difference",
        },
        {
            "test": "Mann-Whitney U",
            "statistic": mw.statistic,
            "p_value": mw.pvalue,
            "interpretation": "location differs" if mw.pvalue < 0.05 else "no clear location difference",
        },
        {
            "test": "Brown-Forsythe",
            "statistic": bf.statistic,
            "p_value": bf.pvalue,
            "interpretation": "variance differs" if bf.pvalue < 0.05 else "no clear variance difference",
        },
        {
            "test": "Wasserstein distance",
            "statistic": wasserstein,
            "p_value": np.nan,
            "interpretation": "distance metric; no p-value",
        },
    ]
    if not math.isnan(js_distance):
        rows.append(
            {
                "test": "Jensen-Shannon distance",
                "statistic": js_distance,
                "p_value": np.nan,
                "interpretation": "distance metric; no p-value",
            }
        )
    return pd.DataFrame(rows)


def _pinball_loss(y_true: pd.Series, y_pred: pd.Series, alpha: float) -> pd.Series:
    diff = y_true - y_pred
    return pd.Series(np.maximum(alpha * diff, (alpha - 1.0) * diff), index=y_true.index)


def _model_performance(predictions: pd.DataFrame, validation_thresholds: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, split), group in predictions.groupby(["model", "split"], sort=False):
        row: dict[str, object] = {"model": model, "split": split, "n_rows": len(group)}
        losses = []
        upper_mask = group["y_true"] > validation_thresholds["p95"]
        lower_mask = group["y_true"] < validation_thresholds["p05"]
        for q_col, alpha in QUANTILE_LEVELS.items():
            loss = _pinball_loss(group["y_true"], group[q_col], alpha)
            row[f"mean_pinball_loss_{q_col}"] = loss.mean()
            losses.append(loss)
        all_pinball = pd.concat(losses, axis=0)
        row["mean_pinball_loss_avg_quantiles"] = all_pinball.mean()
        row["mae_p50"] = (group["p50"] - group["y_true"]).abs().mean()
        row["mbe_p50"] = (group["p50"] - group["y_true"]).mean()
        row["overprediction_share_p50"] = (group["p50"] > group["y_true"]).mean()
        row["underprediction_share_p50"] = (group["p50"] < group["y_true"]).mean()
        row["p10_coverage"] = (group["y_true"] <= group["p10"]).mean()
        row["p30_coverage"] = (group["y_true"] <= group["p30"]).mean()
        row["p50_coverage"] = (group["y_true"] <= group["p50"]).mean()
        row["p70_coverage"] = (group["y_true"] <= group["p70"]).mean()
        row["p90_coverage"] = (group["y_true"] <= group["p90"]).mean()
        row["mean_p50_forecast"] = group["p50"].mean()
        row["mean_realized"] = group["y_true"].mean()
        row["upper_tail_row_share"] = upper_mask.mean()
        row["lower_tail_row_share"] = lower_mask.mean()
        if upper_mask.any():
            upper_losses = [_pinball_loss(group.loc[upper_mask, "y_true"], group.loc[upper_mask, q], a) for q, a in QUANTILE_LEVELS.items()]
            row["upper_tail_mean_pinball_loss_avg_quantiles"] = pd.concat(upper_losses).mean()
            row["upper_tail_pinball_loss_contribution_share"] = pd.concat(upper_losses).sum() / all_pinball.sum()
        else:
            row["upper_tail_mean_pinball_loss_avg_quantiles"] = np.nan
            row["upper_tail_pinball_loss_contribution_share"] = 0.0
        rows.append(row)

    out = pd.DataFrame(rows)
    val = out[out["split"] == "validation"].set_index("model")
    for idx, row in out.iterrows():
        if row["split"] != "test" or row["model"] not in val.index:
            continue
        for metric in ["mean_pinball_loss_avg_quantiles", "mae_p50", "mbe_p50"]:
            denom = val.loc[row["model"], metric]
            if pd.notna(denom) and denom != 0:
                out.loc[idx, f"{metric}_test_to_validation_ratio"] = row[metric] / denom
    model_order = {m: i for i, m in enumerate(MODEL_ORDER)}
    split_order = {"validation": 0, "test": 1}
    out["_model_order"] = out["model"].map(model_order).fillna(99)
    out["_split_order"] = out["split"].map(split_order).fillna(99)
    return out.sort_values(["_model_order", "_split_order"]).drop(columns=["_model_order", "_split_order"])


def _tail_diagnostics(targets: pd.DataFrame, validation_values: pd.Series) -> pd.DataFrame:
    thresholds = {
        "upper_p95": validation_values.quantile(0.95),
        "upper_p99": validation_values.quantile(0.99),
        "lower_p05": validation_values.quantile(0.05),
        "lower_p01": validation_values.quantile(0.01),
    }
    rows: list[dict[str, object]] = []
    for split, group in targets.groupby("split", sort=False):
        s = group["y_true"].dropna()
        for name, threshold in thresholds.items():
            if name.startswith("upper"):
                mask = s > threshold
            else:
                mask = s < threshold
            tail_values = s[mask]
            rows.append(
                {
                    "split": split,
                    "tail_definition": name,
                    "validation_threshold": threshold,
                    "n_events": int(mask.sum()),
                    "event_share": float(mask.mean()),
                    "mean_during_events": tail_values.mean() if len(tail_values) else np.nan,
                    "max_event_value": tail_values.max() if len(tail_values) else np.nan,
                    "min_event_value": tail_values.min() if len(tail_values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _group_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    d = predictions.copy()
    d["hour_of_day"] = d["target_time_utc"].dt.hour
    d["weekday"] = d["target_time_utc"].dt.day_name()
    d["is_weekend"] = d["target_time_utc"].dt.dayofweek >= 5
    d["four_hour_block"] = (d["hour_of_day"] // 4 * 4).map(lambda h: f"{h:02d}:00-{h+3:02d}:00")
    rows: list[dict[str, object]] = []
    for group_name, col in [
        ("lead_hour", "lead_time_h"),
        ("four_hour_block", "four_hour_block"),
        ("hour_of_day", "hour_of_day"),
        ("weekend", "is_weekend"),
    ]:
        for (split, model, value), group in d.groupby(["split", "model", col], sort=False):
            rows.append(
                {
                    "group_type": group_name,
                    "group_value": value,
                    "split": split,
                    "model": model,
                    "count": len(group),
                    "mean_realized": group["y_true"].mean(),
                    "std_realized": group["y_true"].std(ddof=1),
                    "p95_realized": group["y_true"].quantile(0.95),
                    "mae_p50": (group["p50"] - group["y_true"]).abs().mean(),
                    "mbe_p50": (group["p50"] - group["y_true"]).mean(),
                }
            )
    return pd.DataFrame(rows)


def _load_tft_fit_diagnostics() -> pd.DataFrame:
    path = REPO_ROOT / "artifacts/benchmark/tensorboard_training_diagnostics/csv/all_models_fit_diagnostics_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    mask = df["target_label"].astype(str).str.contains("capacity price neg", case=False, na=False)
    return df.loc[mask].copy()


def _plot_distribution(targets: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for split, color in [("validation", "#6CB0D6"), ("test", "#0D4A70")]:
        values = targets.loc[targets["split"] == split, "y_true"].dropna()
        ax.hist(values, bins=60, density=True, alpha=0.45, color=color, label=split.capitalize())
    ax.set_xlabel("Negative aFRR capacity price")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_boxplot(targets: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    data = [targets.loc[targets["split"] == split, "y_true"].dropna() for split in ["validation", "test"]]
    bp = ax.boxplot(data, tick_labels=["Validation", "Test"], patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], ["#9EC9E2", "#226E9C"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("Negative aFRR capacity price")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_timeseries(targets: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    d = targets.sort_values("target_time_utc").copy()
    d["rolling_mean_7d"] = d.groupby("split")["y_true"].transform(lambda s: s.rolling(24 * 7, min_periods=24).mean())
    d["rolling_std_7d"] = d.groupby("split")["y_true"].transform(lambda s: s.rolling(24 * 7, min_periods=24).std())
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.4), sharex=False)
    for ax, split, color in zip(axes, ["validation", "test"], ["#6CB0D6", "#0D4A70"]):
        g = d[d["split"] == split]
        ax.plot(g["target_time_utc"], g["y_true"], color=color, alpha=0.35, linewidth=0.7, label="Realized")
        ax.plot(g["target_time_utc"], g["rolling_mean_7d"], color="#8A1C7C", linewidth=1.1, label="7-day rolling mean")
        ax.plot(g["target_time_utc"], g["rolling_std_7d"], color="#4C4C4C", linewidth=1.1, label="7-day rolling std.")
        ax.set_title(split.capitalize(), loc="left", fontsize=11)
        ax.set_ylabel("Price")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_tft_forecast_distribution(predictions: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    tft = predictions[predictions["model"] == "TFT"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), sharey=True)
    for ax, split in zip(axes, ["validation", "test"]):
        g = tft[tft["split"] == split]
        ax.hist(g["y_true"], bins=60, density=True, alpha=0.45, color="#6CB0D6", label="Realized")
        ax.hist(g["p50"], bins=60, density=True, alpha=0.45, color="#8A1C7C", label="TFT p50")
        ax.set_title(split.capitalize(), fontsize=11)
        ax.set_xlabel("Negative capacity price")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _copy_to_thesis(out_root: Path) -> Path | None:
    if not THESIS_ROOT.exists():
        return None
    dest = THESIS_ROOT / THESIS_REL_ROOT
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(out_root, dest)
    return dest


def _write_interpretation(
    path: Path,
    distribution: pd.DataFrame,
    tests: pd.DataFrame,
    performance: pd.DataFrame,
    fit: pd.DataFrame,
) -> None:
    val = distribution.set_index("split").loc["validation"]
    test = distribution.set_index("split").loc["test"]
    ks_p = tests.loc[tests["test"] == "Kolmogorov-Smirnov", "p_value"].iloc[0]
    bf_p = tests.loc[tests["test"] == "Brown-Forsythe", "p_value"].iloc[0]
    tft_test = performance[(performance["model"] == "TFT") & (performance["split"] == "test")].iloc[0]
    tft_val = performance[(performance["model"] == "TFT") & (performance["split"] == "validation")].iloc[0]

    text = [
        "Observation:",
        (
            f"The validation set covers {int(val['n']):,} hourly observations and the test set covers "
            f"{int(test['n']):,}. The test mean is {test['mean']:.2f} versus {val['mean']:.2f} in validation; "
            f"the test standard deviation is {test['std']:.2f} versus {val['std']:.2f}. "
            f"The validation p95 threshold is exceeded in {test['share_above_validation_p95']:.1%} of test hours "
            f"and the validation p99 threshold is exceeded in {test['share_above_validation_p99']:.1%} of test hours."
        ),
        "",
        "Interpretation:",
        (
            f"The Kolmogorov-Smirnov p-value is {ks_p:.3g} and the Brown-Forsythe p-value is {bf_p:.3g}. "
            "These diagnostics indicate a structural validation-test difference when the p-values are small, "
            "especially if the test period also has higher dispersion or heavier tails. "
            f"For TFT, p50 MAE rises from {tft_val['mae_p50']:.2f} in validation to {tft_test['mae_p50']:.2f} "
            f"in test, while p50 MBE is {tft_test['mbe_p50']:.2f} in test. "
            "This supports the interpretation that test-period degradation is connected to both distribution shift "
            "and model-specific bias under the shifted regime."
        ),
        "",
        "TFT-specific diagnostic:",
    ]
    if not fit.empty:
        row = fit[fit["model"] == "TFT"].iloc[0]
        text.append(
            f"The existing training-fit diagnostics classify TFT as {row['fit_diagnosis']} for this target, "
            f"with a test-to-validation MAE ratio of {row['test_to_validation_mae_ratio']:.2f} and "
            f"validation-loss drift of {row['validation_drift_pct']:.1f}% after the best epoch."
        )
    else:
        text.append("No TensorBoard training-fit diagnostic row was found for TFT in the existing exports.")
    text.extend(
        [
            "",
            "Limitation:",
            (
                "The analysis is diagnostic rather than causal proof. The two-sample tests treat observations as if "
                "they were independent, while electricity market time series contain serial dependence, repeated "
                "lead-time rows and regime persistence."
            ),
        ]
    )
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    dirs = _ensure_dirs(OUT_ROOT)
    targets_by_split = {split: _load_target_split(split) for split in ["validation", "test"]}
    hourly_targets = pd.concat(targets_by_split.values(), ignore_index=True)
    predictions, mapping_log = _load_predictions(targets_by_split)

    validation_values = targets_by_split["validation"]["y_true"]
    thresholds = {
        "p01": validation_values.quantile(0.01),
        "p05": validation_values.quantile(0.05),
        "p95": validation_values.quantile(0.95),
        "p99": validation_values.quantile(0.99),
    }
    distribution = _distribution_summary(hourly_targets, validation_values)
    shift_tests = _shift_tests(targets_by_split["validation"]["y_true"], targets_by_split["test"]["y_true"])
    tails = _tail_diagnostics(hourly_targets, validation_values)
    performance = _model_performance(predictions, thresholds)
    groups = _group_diagnostics(predictions)
    fit = _load_tft_fit_diagnostics()

    mapping_log.to_csv(dirs["diagnostics"] / "input_mapping_log.csv", index=False)
    distribution.to_csv(dirs["csv"] / "validation_test_distribution_summary.csv", index=False)
    shift_tests.to_csv(dirs["csv"] / "validation_test_shift_tests.csv", index=False)
    tails.to_csv(dirs["csv"] / "validation_test_tail_spike_diagnostics.csv", index=False)
    performance.to_csv(dirs["csv"] / "validation_test_model_performance_degradation.csv", index=False)
    groups.to_csv(dirs["csv"] / "validation_test_block_lead_hour_diagnostics.csv", index=False)
    fit.to_csv(dirs["csv"] / "capacity_price_neg_training_fit_diagnostics.csv", index=False)

    dist_latex = distribution.copy()
    dist_latex["split"] = dist_latex["split"].str.capitalize()
    _write_latex_table(
        dist_latex,
        dirs["tables"] / "validation_test_distribution_summary.tex",
        caption="Validation-test distribution comparison for negative aFRR capacity prices.",
        label="tab:rq1-capacity-price-neg-validation-test-distribution",
        columns=["split", "n", "mean", "median", "std", "p05", "p95", "p99", "share_above_validation_p95"],
    )
    _write_latex_table(
        shift_tests,
        dirs["tables"] / "validation_test_shift_tests.tex",
        caption="Validation-test distribution-shift diagnostics for negative aFRR capacity prices. P-values are interpreted cautiously because the observations are time series.",
        label="tab:rq1-capacity-price-neg-validation-test-shift-tests",
        columns=["test", "statistic", "p_value", "interpretation"],
    )
    perf_compact = performance[
        [
            "model",
            "split",
            "mean_pinball_loss_avg_quantiles",
            "mae_p50",
            "mbe_p50",
            "overprediction_share_p50",
            "p50_coverage",
            "upper_tail_pinball_loss_contribution_share",
        ]
    ].copy()
    perf_compact["split"] = perf_compact["split"].str.capitalize()
    _write_latex_table(
        perf_compact,
        dirs["tables"] / "validation_test_model_performance_degradation.tex",
        caption="Validation-test performance degradation for negative aFRR capacity prices. The table compares forecast errors and bias metrics across models.",
        label="tab:rq1-capacity-price-neg-validation-test-model-degradation",
        columns=[
            "model",
            "split",
            "mean_pinball_loss_avg_quantiles",
            "mae_p50",
            "mbe_p50",
            "overprediction_share_p50",
            "p50_coverage",
            "upper_tail_pinball_loss_contribution_share",
        ],
    )

    _plot_distribution(hourly_targets, dirs["figures"] / "capacity_price_neg_validation_test_distribution.png")
    _plot_boxplot(hourly_targets, dirs["figures"] / "capacity_price_neg_validation_test_boxplot.png")
    _plot_timeseries(hourly_targets, dirs["figures"] / "capacity_price_neg_validation_test_timeseries.png")
    _plot_tft_forecast_distribution(predictions, dirs["figures"] / "capacity_price_neg_tft_forecast_distribution.png")

    _write_latex_figure(
        dirs["latex_figures"] / "capacity_price_neg_validation_test_distribution.tex",
        image_name="capacity_price_neg_validation_test_distribution.png",
        caption="Validation-test distribution comparison for negative aFRR capacity prices. The figure compares the realized target distribution in the validation and test periods to assess whether the test period represents a different price regime.",
        label="fig:rq1-capacity-price-neg-validation-test-distribution",
    )
    _write_latex_figure(
        dirs["latex_figures"] / "capacity_price_neg_validation_test_boxplot.tex",
        image_name="capacity_price_neg_validation_test_boxplot.png",
        caption="Validation-test boxplot comparison for negative aFRR capacity prices. Outliers are hidden to show the central distribution.",
        label="fig:rq1-capacity-price-neg-validation-test-boxplot",
        width=r"0.75\linewidth",
    )
    _write_latex_figure(
        dirs["latex_figures"] / "capacity_price_neg_validation_test_timeseries.tex",
        image_name="capacity_price_neg_validation_test_timeseries.png",
        caption="Validation-test time-series diagnostics for negative aFRR capacity prices. Lines show realized values, seven-day rolling means and seven-day rolling standard deviations.",
        label="fig:rq1-capacity-price-neg-validation-test-timeseries",
    )
    _write_latex_figure(
        dirs["latex_figures"] / "capacity_price_neg_tft_forecast_distribution.tex",
        image_name="capacity_price_neg_tft_forecast_distribution.png",
        caption="TFT forecast-distribution diagnostic for negative aFRR capacity prices. The figure compares realized values with TFT p50 forecasts in validation and test rows.",
        label="fig:rq1-capacity-price-neg-tft-forecast-distribution",
    )

    _write_interpretation(
        dirs["text"] / "capacity_price_neg_validation_test_shift_interpretation.txt",
        distribution,
        shift_tests,
        performance,
        fit,
    )

    thesis_dest = _copy_to_thesis(OUT_ROOT)
    print("[OK] Negative aFRR capacity price validation-test diagnostics written.")
    print(f"[OUT] {OUT_ROOT}")
    if thesis_dest:
        print(f"[OUT] {thesis_dest}")
    print("\nDistribution summary:")
    print(distribution.to_string(index=False))
    print("\nShift tests:")
    print(shift_tests.to_string(index=False))
    print("\nModel performance:")
    print(performance[["model", "split", "mean_pinball_loss_avg_quantiles", "mae_p50", "mbe_p50", "overprediction_share_p50"]].to_string(index=False))


if __name__ == "__main__":
    main()
