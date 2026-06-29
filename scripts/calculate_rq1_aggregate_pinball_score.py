#!/usr/bin/env python3
"""Calculate Aggregate Pinball Score from existing RQ1 joined forecasts.

This script is intentionally read-only with respect to forecast/model outputs. It
loads the joined test prediction parquet files, computes pinball loss over all
available quantile columns, and writes audit tables for thesis reporting.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MODEL_LABELS = {"linear": "RLQR", "xgb": "XGB", "tft": "TFT"}
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
    "pred_afrr_capacity_price_neg": r"aFRR capacity price $-$",
    "pred_afrr_activation_price_pos": "aFRR activation price +",
    "pred_afrr_activation_price_neg": r"aFRR activation price $-$",
    "pred_afrr_activation_rate_pos": "aFRR activation rate +",
    "pred_afrr_activation_rate_neg": r"aFRR activation rate $-$",
}


@dataclass(frozen=True)
class ApsRecord:
    model: str
    model_key: str
    target: str
    target_label: str
    aps: float
    n_obs: int
    n_pinball_terms: int
    n_quantiles: int
    quantiles_used: str
    n_removed_missing: int


def _parse_quantile_column(column: str) -> float | None:
    match = re.fullmatch(r"p(\d{1,2})", column.strip().lower())
    if not match:
        return None
    value = int(match.group(1))
    if value <= 0 or value >= 100:
        return None
    return value / 100.0


def _target_sort_key(target: str) -> int:
    try:
        return TARGET_ORDER.index(target)
    except ValueError:
        return len(TARGET_ORDER)


def _model_sort_key(model: str) -> int:
    try:
        return MODEL_ORDER.index(model)
    except ValueError:
        return len(MODEL_ORDER)


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    if abs(value) < 0.01 and value != 0:
        return f"{value:.5f}"
    return f"{value:.2f}"


def _latex_escape(value: str) -> str:
    return (
        str(value)
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def _pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, alpha: float) -> np.ndarray:
    diff = y_true - q_pred
    return np.where(diff >= 0.0, alpha * diff, (1.0 - alpha) * (-diff))


def _iter_prediction_files(input_dir: Path) -> Iterable[Path]:
    return sorted(input_dir.glob("*__test__*.parquet"))


def calculate_aps(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    records: list[ApsRecord] = []
    quantile_records: list[dict[str, object]] = []
    warnings: list[str] = []

    files = list(_iter_prediction_files(input_dir))
    if not files:
        raise FileNotFoundError(f"No joined prediction parquet files found in {input_dir}")

    for path in files:
        parts = path.stem.split("__test__")
        if len(parts) != 2:
            warnings.append(f"Skipping unexpected file name: {path.name}")
            continue
        model_key = parts[0]
        model = MODEL_LABELS.get(model_key, model_key.upper())
        target = parts[1]
        target_label = TARGET_LABELS.get(target, target)

        data = pd.read_parquet(path)
        required = {"y_true", "target_time_utc", "snapshot_time_utc", "lead_time_h"}
        missing_required = sorted(required.difference(data.columns))
        if missing_required:
            raise ValueError(f"{path.name} is missing required columns: {missing_required}")

        quantile_pairs = sorted(
            ((col, alpha) for col in data.columns if (alpha := _parse_quantile_column(col)) is not None),
            key=lambda item: item[1],
        )
        if not quantile_pairs:
            raise ValueError(f"{path.name} contains no parseable quantile columns such as p10.")

        keys = ["snapshot_time_utc", "target_time_utc", "lead_time_h"]
        duplicate_count = int(data.duplicated(keys).sum())
        if duplicate_count:
            raise ValueError(f"{path.name} has {duplicate_count} duplicated forecast rows by {keys}.")

        y = pd.to_numeric(data["y_true"], errors="coerce")
        all_losses: list[np.ndarray] = []
        n_removed_missing_total = 0
        quantiles_used: list[str] = []
        n_obs_reference: int | None = None

        for col, alpha in quantile_pairs:
            q = pd.to_numeric(data[col], errors="coerce")
            valid = y.notna() & q.notna()
            n_removed = int((~valid).sum())
            n_removed_missing_total += n_removed
            if n_removed:
                warnings.append(f"{model} {target} {col}: removed {n_removed} rows with missing y_true or forecast.")
            if not bool(valid.any()):
                raise ValueError(f"{model} {target} {col}: no aligned non-missing rows remain.")

            y_values = y.loc[valid].to_numpy(dtype=float)
            q_values = q.loc[valid].to_numpy(dtype=float)
            if np.array_equal(y_values, q_values):
                warnings.append(f"{model} {target} {col}: forecast values exactly equal realized values for all rows.")

            losses = _pinball_loss(y_values, q_values, alpha)
            if not np.isfinite(losses).all():
                raise ValueError(f"{model} {target} {col}: non-finite pinball loss encountered.")
            min_loss = float(np.min(losses))
            if min_loss < -1e-10:
                raise ValueError(f"{model} {target} {col}: negative pinball loss encountered ({min_loss}).")
            losses = np.maximum(losses, 0.0)
            all_losses.append(losses)
            quantiles_used.append(col)

            if n_obs_reference is None:
                n_obs_reference = int(valid.sum())
            elif n_obs_reference != int(valid.sum()):
                warnings.append(
                    f"{model} {target}: quantile {col} has {int(valid.sum())} valid rows; "
                    f"first quantile had {n_obs_reference}."
                )

            quantile_records.append(
                {
                    "model": model,
                    "model_key": model_key,
                    "target": target,
                    "target_label": target_label,
                    "quantile": col,
                    "alpha": alpha,
                    "mean_pinball_loss": float(np.mean(losses)),
                    "n_obs": int(valid.sum()),
                    "n_removed_missing": n_removed,
                }
            )

        concatenated = np.concatenate(all_losses)
        aps = float(np.mean(concatenated))
        quantile_mean_average = float(np.mean([float(np.mean(loss)) for loss in all_losses]))
        if not math.isclose(aps, quantile_mean_average, rel_tol=1e-12, abs_tol=1e-12):
            warnings.append(
                f"{model} {target}: APS differs from equal-quantile mean because valid row counts differ."
            )
        records.append(
            ApsRecord(
                model=model,
                model_key=model_key,
                target=target,
                target_label=target_label,
                aps=aps,
                n_obs=int(n_obs_reference or 0),
                n_pinball_terms=int(sum(len(loss) for loss in all_losses)),
                n_quantiles=len(quantile_pairs),
                quantiles_used=",".join(quantiles_used),
                n_removed_missing=n_removed_missing_total,
            )
        )

    target_df = pd.DataFrame([record.__dict__ for record in records])
    if target_df.empty:
        raise ValueError("No APS records were computed.")

    target_df["_target_order"] = target_df["target"].map(_target_sort_key)
    target_df["_model_order"] = target_df["model"].map(_model_sort_key)
    target_df = target_df.sort_values(["_target_order", "_model_order"]).drop(columns=["_target_order", "_model_order"])
    quantile_df = pd.DataFrame(quantile_records).sort_values(["target", "model", "alpha"])

    quantile_sets = (
        target_df.groupby(["model", "target"], as_index=False)["quantiles_used"]
        .first()
        .groupby("quantiles_used")
        .size()
    )
    if len(quantile_sets) > 1:
        warnings.append("Quantile sets differ across model-target combinations: " + "; ".join(f"{k}={v}" for k, v in quantile_sets.items()))

    rlqr = target_df.loc[target_df["model"].eq("RLQR"), ["target", "aps"]].rename(columns={"aps": "rlqr_aps"})
    target_df = target_df.merge(rlqr, on="target", how="left")
    invalid_rlqr = target_df["rlqr_aps"].isna() | ~np.isfinite(target_df["rlqr_aps"]) | (target_df["rlqr_aps"] <= 0)
    if bool(invalid_rlqr.any()):
        bad_targets = sorted(target_df.loc[invalid_rlqr, "target"].unique())
        raise ValueError(f"Cannot compute RLQR-normalized APS for targets with invalid RLQR APS: {bad_targets}")
    target_df["relative_aps_to_rlqr"] = target_df["aps"] / target_df["rlqr_aps"]
    target_df = target_df.drop(columns=["rlqr_aps"])

    pooled = (
        target_df.groupby("model", as_index=False)
        .agg(
            pooled_raw_aps_observation_weighted=("aps", lambda s: np.average(s, weights=target_df.loc[s.index, "n_pinball_terms"])),
            pooled_raw_aps_target_mean=("aps", "mean"),
            pooled_relative_aps_to_rlqr=("relative_aps_to_rlqr", "mean"),
            n_targets=("target", "nunique"),
            n_pinball_terms=("n_pinball_terms", "sum"),
        )
        .sort_values("model", key=lambda s: s.map(_model_sort_key))
    )
    return target_df, quantile_df, pooled, warnings


def write_latex_target_table(path: Path, target_df: pd.DataFrame) -> None:
    pivot_aps = target_df.pivot(index="target_label", columns="model", values="aps")
    pivot_rel = target_df.pivot(index="target_label", columns="model", values="relative_aps_to_rlqr")
    labels = target_df[["target", "target_label"]].drop_duplicates().sort_values("target", key=lambda s: s.map(_target_sort_key))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Aggregate Pinball Score by forecast target and model. APS averages pinball loss across the evaluated quantiles and aligned test observations; lower values indicate stronger probabilistic forecast performance.}",
        r"\label{tab:aggregate_pinball_score_by_target_model}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Forecast target} & \textbf{RLQR} & \textbf{XGB} & \textbf{TFT} & \textbf{Best model} \\",
        r"\midrule",
    ]
    for _, row in labels.iterrows():
        label = row["target_label"]
        values = {model: float(pivot_aps.loc[label, model]) for model in MODEL_ORDER if model in pivot_aps.columns}
        best = min(values, key=values.get)
        cells = []
        for model in MODEL_ORDER:
            val = values.get(model, math.nan)
            formatted = _format_float(val)
            if model == best:
                formatted = rf"\textbf{{{formatted}}}"
            cells.append(formatted)
        lines.append(f"{label} & " + " & ".join(cells) + f" & {best} " + r"\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_pooled_table(path: Path, pooled_df: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Pooled Aggregate Pinball Score summary by model. Raw pooled APS is scale-dependent; the relative APS column averages target-wise APS normalized to RLQR.}",
        r"\label{tab:aggregate_pinball_score_pooled_model}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{\shortstack{Raw APS\\obs.-weighted}} & \textbf{\shortstack{Raw APS\\target mean}} & \textbf{\shortstack{Relative APS\\to RLQR}} \\",
        r"\midrule",
    ]
    for _, row in pooled_df.iterrows():
        lines.append(
            f"{_latex_escape(row['model'])} & "
            f"{_format_float(float(row['pooled_raw_aps_observation_weighted']))} & "
            f"{_format_float(float(row['pooled_raw_aps_target_mean']))} & "
            f"{float(row['pooled_relative_aps_to_rlqr']):.3f} "
            r"\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/benchmark/rq1_ml_model_benchmark/diagnostics/joined_predictions"),
        help="Directory containing joined prediction parquet files.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("artifacts/benchmark/rq1_ml_model_benchmark/result_section"),
        help="Output root containing csv/ and latex_tables/ subdirectories.",
    )
    args = parser.parse_args()

    target_df, quantile_df, pooled_df, warnings = calculate_aps(args.input_dir)
    csv_dir = args.out_root / "csv"
    table_dir = args.out_root / "latex_tables"
    csv_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    target_csv = csv_dir / "aggregate_pinball_score_by_target_model.csv"
    quantile_csv = csv_dir / "aggregate_pinball_score_per_quantile.csv"
    pooled_csv = csv_dir / "aggregate_pinball_score_pooled_by_model.csv"
    target_df.to_csv(target_csv, index=False)
    quantile_df.to_csv(quantile_csv, index=False)
    pooled_df.to_csv(pooled_csv, index=False)

    target_tex = table_dir / "aggregate_pinball_score_by_target_model.tex"
    pooled_tex = table_dir / "aggregate_pinball_score_pooled_by_model.tex"
    write_latex_target_table(target_tex, target_df)
    write_latex_pooled_table(pooled_tex, pooled_df)

    print("Aggregate Pinball Score by target: best model")
    for target in TARGET_ORDER:
        group = target_df.loc[target_df["target"].eq(target)]
        if group.empty:
            continue
        best = group.loc[group["aps"].idxmin()]
        print(f"  - {TARGET_LABELS.get(target, target)}: {best['model']} APS={_format_float(float(best['aps']))}")

    print("\nPooled APS by model")
    print(pooled_df.to_string(index=False, formatters={
        "pooled_raw_aps_observation_weighted": lambda x: _format_float(float(x)),
        "pooled_raw_aps_target_mean": lambda x: _format_float(float(x)),
        "pooled_relative_aps_to_rlqr": lambda x: f"{float(x):.3f}",
    }))

    quantile_sets = target_df[["model", "target", "quantiles_used"]].drop_duplicates()
    print("\nQuantiles used:")
    for quantiles, rows in quantile_sets.groupby("quantiles_used"):
        print(f"  - {quantiles}: {len(rows)} model-target combinations")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    print("\nWrote:")
    for path in [target_csv, quantile_csv, pooled_csv, target_tex, pooled_tex]:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
