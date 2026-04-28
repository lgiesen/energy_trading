#!/usr/bin/env python3
"""Pre-simulation trading-relevance scorecard across evaluated model runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_trading.visualization.style import apply_geo_style, get_color  # noqa: E402


def _configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-paper")
    sns.set_palette([get_color("primary"), get_color("secondary"), get_color("tertiary"), get_color("neutral_dark")])
    apply_geo_style()
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": True,
            "grid.alpha": 0.3,
        }
    )


def _save_plot(fig: plt.Figure, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(v: object) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return float("nan")


def _load_summary_rows(run_dir: Path, expected_split: str) -> pd.DataFrame:
    p = run_dir / "summary_metrics.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing summary_metrics.json in {run_dir}. Run evaluate_individual_run.py first.")
    payload = _read_json(p)
    split_observed = str(payload.get("split", "")).strip().lower()
    if split_observed and split_observed != expected_split:
        raise ValueError(
            f"Split mismatch for {run_dir}: summary_metrics.json has split='{split_observed}', "
            f"expected '{expected_split}'."
        )
    rows = payload.get("avg_metrics_by_prediction_column", [])
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No avg_metrics_by_prediction_column rows in {p}")
    return df


def _load_gate_time_rows(run_dir: Path, expected_split: str) -> pd.DataFrame:
    p = run_dir / f"{expected_split}_gate_time_metrics_by_target.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    if d.empty:
        return d
    d["prediction_column"] = d["prediction_column"].astype(str)
    return d


def _load_spread_metrics(run_dir: Path, expected_split: str) -> dict[str, Any]:
    p = run_dir / f"{expected_split}_gate_time_spread_metrics.json"
    if not p.exists():
        return {}
    try:
        return _read_json(p)
    except Exception:
        return {}


def _load_context_checksums(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "training_run_context.json"
    if not p.exists():
        return {}
    payload = _read_json(p)
    return payload.get("input_checksums", {})


def _score_row(row: pd.Series) -> dict[str, float]:
    gate_dplus1_mae = _safe_float(row.get("gate_dplus1_mae"))
    gate_mae = _safe_float(row.get("gate_mae"))
    mae = _safe_float(row.get("mae_mean"))
    err_ref = gate_dplus1_mae if np.isfinite(gate_dplus1_mae) else (gate_mae if np.isfinite(gate_mae) else mae)
    err_score = 1.0 / (1.0 + max(0.0, err_ref)) if np.isfinite(err_ref) else 0.0

    dir_acc = _safe_float(row.get("gate_dplus1_directional_accuracy"))
    if not np.isfinite(dir_acc):
        dir_acc = _safe_float(row.get("directional_accuracy"))
    dir_score = float(np.clip(dir_acc, 0.0, 1.0)) if np.isfinite(dir_acc) else 0.5

    spread_err = _safe_float(row.get("spread_directional_error"))
    spread_score = float(np.clip(1.0 - spread_err, 0.0, 1.0)) if np.isfinite(spread_err) else 0.5

    spike_mae = _safe_float(row.get("spike_mae_top5_h1"))
    spike_score = 1.0 / (1.0 + max(0.0, spike_mae)) if np.isfinite(spike_mae) else err_score

    # Global sanity check: penalize catastrophic global behavior without dominating gate-time score.
    global_err_score = 1.0 / (1.0 + max(0.0, mae)) if np.isfinite(mae) else 0.0
    sanity_penalty = 0.0
    if np.isfinite(mae) and np.isfinite(gate_dplus1_mae) and mae > 3.0 * max(gate_dplus1_mae, 1e-9):
        sanity_penalty = 0.05

    # Gate-time-first composite.
    composite = 0.55 * err_score + 0.25 * spread_score + 0.10 * dir_score + 0.05 * spike_score + 0.05 * global_err_score
    composite = max(0.0, composite - sanity_penalty)
    return {
        "score_err_component": err_score,
        "score_spread_component": spread_score,
        "score_dir_component": dir_score,
        "score_spike_component": spike_score,
        "score_global_sanity_component": global_err_score,
        "score_sanity_penalty": sanity_penalty,
        "score_trading_relevance": composite,
    }


def _protocol_audit(
    run_dirs: list[Path],
    labels: list[str],
    score_df: pd.DataFrame,
    split: str,
) -> dict[str, Any]:
    checksum_rows = []
    for rd, lbl in zip(run_dirs, labels):
        checks = _load_context_checksums(rd)
        da_sha = (
            checks.get("da_base_dir", {}).get("feature_config_sha256")
            if isinstance(checks, dict)
            else None
        )
        afrr_sha = (
            checks.get("afrr_base_dir", {}).get("feature_config_sha256")
            if isinstance(checks, dict)
            else None
        )
        checksum_rows.append({"label": lbl, "run_dir": str(rd), "da_feature_config_sha256": da_sha, "afrr_feature_config_sha256": afrr_sha})

    chk_df = pd.DataFrame(checksum_rows)
    da_unique = sorted({x for x in chk_df.get("da_feature_config_sha256", pd.Series(dtype=str)).dropna().astype(str).tolist() if x})
    afrr_unique = sorted({x for x in chk_df.get("afrr_feature_config_sha256", pd.Series(dtype=str)).dropna().astype(str).tolist() if x})

    pred_sets = {
        lbl: set(score_df.loc[score_df["model_label"] == lbl, "prediction_column"].astype(str).tolist())
        for lbl in labels
    }
    if pred_sets:
        common_preds = sorted(set.intersection(*pred_sets.values()))
        union_preds = sorted(set.union(*pred_sets.values()))
    else:
        common_preds, union_preds = [], []

    gate_present = False
    if "gate_dplus1_mae" in score_df.columns:
        gate_present = bool(pd.to_numeric(score_df["gate_dplus1_mae"], errors="coerce").notna().any())
    elif "gate_mae" in score_df.columns:
        gate_present = bool(pd.to_numeric(score_df["gate_mae"], errors="coerce").notna().any())

    return {
        "split_used_for_comparison": split,
        "runs": checksum_rows,
        "same_da_feature_config_across_runs": len(da_unique) <= 1 and len(da_unique) > 0,
        "same_afrr_feature_config_across_runs": len(afrr_unique) <= 1 and len(afrr_unique) > 0,
        "common_prediction_columns": common_preds,
        "union_prediction_columns": union_preds,
        "prediction_column_sets_equal": len(common_preds) == len(union_preds) and len(union_preds) > 0,
        "gate_metrics_present_for_some_targets": gate_present,
        "notes": [
            "Use identical split=test for final claims, split=val for tuning.",
            "Runs should be trained from same feature snapshot/checksum.",
            "Compare only on common prediction columns.",
        ],
    }


def _plot_score_heatmap(df: pd.DataFrame, out_png: Path) -> None:
    if df.empty:
        return
    piv = df.pivot_table(
        index="prediction_column",
        columns="model_label",
        values="score_trading_relevance",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.heatmap(
        piv,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        cbar_kws={"label": "Trading relevance score"},
        ax=ax,
    )
    ax.set_title("Pre-simulation Trading Relevance Score by Target")
    ax.set_xlabel("Model")
    ax.set_ylabel("Prediction column")
    fig.tight_layout()
    _save_plot(fig, out_png)


def _plot_target_ranking(df: pd.DataFrame, out_png: Path) -> None:
    if df.empty:
        return
    d = df.copy().sort_values(["prediction_column", "score_trading_relevance"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=d,
        x="prediction_column",
        y="score_trading_relevance",
        hue="model_label",
        ax=ax,
    )
    ax.set_title("Model Ranking per Target (Pre-simulation score)")
    ax.set_xlabel("Prediction column")
    ax.set_ylabel("Trading relevance score")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    _save_plot(fig, out_png)


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build pre-simulation scorecard and ranking by target.")
    p.add_argument("--run-dirs", nargs="+", required=True, help="Evaluated run directories.")
    p.add_argument("--labels", nargs="+", required=True, help="Model labels, same length as --run-dirs.")
    p.add_argument("--split", choices=["val", "test"], default="test", help="Comparison split (must match evaluated summaries).")
    p.add_argument("--out-dir", default="artifacts/benchmarks/pre_sim_scorecard", help="Output directory.")
    return p


def main() -> None:
    args = _build_cli().parse_args()
    run_dirs = [Path(p) for p in args.run_dirs]
    labels = args.labels
    if len(run_dirs) != len(labels):
        raise ValueError("--run-dirs and --labels must have identical length.")

    _configure_plot_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    spread_rows: list[dict[str, Any]] = []
    for rd, lbl in zip(run_dirs, labels):
        sdf = _load_summary_rows(rd, expected_split=args.split)
        gdf = _load_gate_time_rows(rd, expected_split=args.split)
        if not gdf.empty:
            # Primary strict metric per target: protocol strict_dplus1 at target-specific default gate.
            gsmall = (
                gdf[gdf["protocol"].astype(str) == "strict_dplus1"]
                .sort_values(["prediction_column", "n_rows_gate_dplus1"], ascending=[True, False])
                .groupby("prediction_column", as_index=False)
                .head(1)
            )
            keep = [
                "prediction_column",
                "gate_dplus1_mae",
                "gate_dplus1_rmse",
                "gate_dplus1_mbe",
                "gate_dplus1_directional_accuracy",
                "n_rows_gate_dplus1",
            ]
            gsmall = gsmall[[c for c in keep if c in gsmall.columns]].drop_duplicates(subset=["prediction_column"])
            sdf = sdf.merge(gsmall, on="prediction_column", how="left")
        sdf["model_label"] = lbl
        sdf["run_dir"] = str(rd.resolve())
        frames.append(sdf)
        spread = _load_spread_metrics(rd, expected_split=args.split)
        spread_rows.append(
            {
                "model_label": lbl,
                "run_dir": str(rd.resolve()),
                "spread_directional_error": _safe_float(spread.get("spread_directional_error")),
                "spread_directional_accuracy": _safe_float(spread.get("spread_directional_accuracy")),
                "spread_n_rows": _safe_float(spread.get("n_rows")),
            }
        )
    all_df = pd.concat(frames, axis=0, ignore_index=True)
    spread_df = pd.DataFrame(spread_rows)
    if not spread_df.empty:
        all_df = all_df.merge(spread_df[["model_label", "spread_directional_error", "spread_directional_accuracy", "spread_n_rows"]], on="model_label", how="left")

    # Fair comparison: keep common targets only.
    pred_sets = {
        lbl: set(all_df.loc[all_df["model_label"] == lbl, "prediction_column"].astype(str).tolist())
        for lbl in labels
    }
    common_preds = set.intersection(*pred_sets.values()) if pred_sets else set()
    if not common_preds:
        raise ValueError("No common prediction columns across compared runs.")
    all_df = all_df[all_df["prediction_column"].isin(common_preds)].copy()

    score_rows = []
    for _, row in all_df.iterrows():
        rec = row.to_dict()
        rec.update(_score_row(row))
        score_rows.append(rec)
    score_df = pd.DataFrame(score_rows)

    # Rank within each prediction target (higher score is better).
    score_df["rank_within_target"] = score_df.groupby("prediction_column")["score_trading_relevance"].rank(
        ascending=False,
        method="min",
    )

    protocol = _protocol_audit(run_dirs=run_dirs, labels=labels, score_df=score_df, split=args.split)

    score_csv = out_dir / "pre_sim_scorecard.csv"
    score_df.sort_values(["prediction_column", "rank_within_target", "model_label"]).to_csv(score_csv, index=False)

    best_df = (
        score_df.sort_values(["prediction_column", "score_trading_relevance"], ascending=[True, False])
        .groupby("prediction_column", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_csv = out_dir / "pre_sim_best_model_per_target.csv"
    best_df.to_csv(best_csv, index=False)

    protocol_json = out_dir / "comparison_protocol_audit.json"
    protocol_json.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    _plot_score_heatmap(score_df, out_dir / "pre_sim_score_heatmap.png")
    _plot_target_ranking(score_df, out_dir / "pre_sim_target_ranking.png")

    print("[OK] Pre-simulation scorecard created.")
    print(f"- {score_csv}")
    print(f"- {best_csv}")
    print(f"- {protocol_json}")
    print(f"- {out_dir / 'pre_sim_score_heatmap.png'}")
    print(f"- {out_dir / 'pre_sim_target_ranking.png'}")


if __name__ == "__main__":
    main()
