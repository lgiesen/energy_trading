#!/usr/bin/env python3
"""Generate a thesis-ready final benchmark report package.

Outputs:
1) Unified model comparison table for test split.
2) Recommendation per target.
3) Explicit open-risk list before simulation backtest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RECOMMENDED_METRICS = [
    "mae_mean",
    "rmse_mean",
    "wmape_mean",
    "mbe_mean",
    "over_prediction_ratio_mean",
    "gate_mae",
    "gate_rmse",
    "directional_accuracy",
    "skill_score_mae_mean",
    "spike_mae_top5_h1",
    "spike_recall_top5_h1",
    "lag_best_h_h1",
    "gate_dplus1_mae",
    "gate_dplus1_rmse",
    "gate_dplus1_mbe",
    "gate_dplus1_directional_accuracy",
    "spread_directional_error",
]

TARGET_FAMILY_ORDER = [
    "pred_da_price",
    "pred_afrr_capacity_price_pos",
    "pred_afrr_capacity_price_neg",
    "pred_afrr_activation_price_pos",
    "pred_afrr_activation_price_neg",
    "pred_afrr_activation_rate_pos",
    "pred_afrr_activation_rate_neg",
]


@dataclass(frozen=True)
class ModelInput:
    label: str
    run_dir: Path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def _score_row(row: pd.Series) -> float:
    # Trading-relevant composite score:
    # gate-aware error (or MAE fallback) + direction + spike + skill.
    gate_mae = _safe_float(row.get("gate_mae"))
    mae = _safe_float(row.get("mae_mean"))
    err = gate_mae if np.isfinite(gate_mae) else mae
    err_score = 1.0 / (1.0 + max(0.0, err)) if np.isfinite(err) else 0.0

    direction = _safe_float(row.get("directional_accuracy"))
    direction = float(np.clip(direction, 0.0, 1.0)) if np.isfinite(direction) else 0.5

    spike_recall = _safe_float(row.get("spike_recall_top5_h1"))
    spike_recall = float(np.clip(spike_recall, 0.0, 1.0)) if np.isfinite(spike_recall) else 0.5

    skill = _safe_float(row.get("skill_score_mae_mean"))
    skill_score = float(np.clip((skill + 1.0) / 2.0, 0.0, 1.0)) if np.isfinite(skill) else 0.5

    return 0.45 * err_score + 0.20 * direction + 0.20 * spike_recall + 0.15 * skill_score


def _score_row_execution(row: pd.Series) -> float:
    gate_dplus1_mae = _safe_float(row.get("gate_dplus1_mae"))
    gate_mae = _safe_float(row.get("gate_mae"))
    mae = _safe_float(row.get("mae_mean"))
    err = gate_dplus1_mae if np.isfinite(gate_dplus1_mae) else (gate_mae if np.isfinite(gate_mae) else mae)
    err_score = 1.0 / (1.0 + max(0.0, err)) if np.isfinite(err) else 0.0

    spread_err = _safe_float(row.get("spread_directional_error"))
    spread_score = float(np.clip(1.0 - spread_err, 0.0, 1.0)) if np.isfinite(spread_err) else 0.5

    direction = _safe_float(row.get("gate_dplus1_directional_accuracy"))
    if not np.isfinite(direction):
        direction = _safe_float(row.get("directional_accuracy"))
    direction = float(np.clip(direction, 0.0, 1.0)) if np.isfinite(direction) else 0.5

    global_sanity = 1.0 / (1.0 + max(0.0, mae)) if np.isfinite(mae) else 0.0
    return 0.60 * err_score + 0.25 * spread_score + 0.10 * direction + 0.05 * global_sanity


def _label_to_expected_family(label: str) -> str | None:
    l = (label or "").strip().lower()
    if "xgb" in l or "xgboost" in l:
        return "xgboost"
    if "tft" in l:
        return "tft"
    if "lin" in l or "ridge" in l:
        return "linear"
    return None


def _detect_run_families(run_dir: Path) -> set[str]:
    pred_dir = run_dir / "predictions"
    if not pred_dir.exists():
        return set()
    fam: set[str] = set()
    for p in pred_dir.glob("*.parquet"):
        n = p.name.lower()
        if n.startswith("xgboost_"):
            fam.add("xgboost")
        elif n.startswith("linear_"):
            fam.add("linear")
        elif n.startswith("da_target_") or n.startswith("afrr_target_"):
            fam.add("tft")
    return fam


def _load_rows(model: ModelInput, expected_split: str) -> pd.DataFrame:
    summary_path = model.run_dir / "summary_metrics.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary_metrics.json: {summary_path}")
    payload = _read_json(summary_path)
    split_observed = str(payload.get("split", "")).strip().lower()
    if split_observed and split_observed != expected_split:
        raise ValueError(
            f"Split mismatch for {model.label}: summary_metrics.json has split='{split_observed}', "
            f"expected '{expected_split}'."
        )
    rows = payload.get("avg_metrics_by_prediction_column", [])
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No avg_metrics_by_prediction_column rows in: {summary_path}")
    df["model_label"] = model.label
    df["run_dir"] = str(model.run_dir.resolve())
    gate_path = model.run_dir / f"{expected_split}_gate_time_metrics_by_target.csv"
    if gate_path.exists():
        gdf = pd.read_csv(gate_path)
        if not gdf.empty:
            gsel = (
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
            ]
            gsel = gsel[[c for c in keep if c in gsel.columns]].drop_duplicates(subset=["prediction_column"])
            df = df.merge(gsel, on="prediction_column", how="left")

    spread_path = model.run_dir / f"{expected_split}_gate_time_spread_metrics.json"
    if spread_path.exists():
        try:
            spread = _read_json(spread_path)
            df["spread_directional_error"] = _safe_float(spread.get("spread_directional_error"))
            df["spread_directional_accuracy"] = _safe_float(spread.get("spread_directional_accuracy"))
        except Exception:
            pass
    return df


def _normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in RECOMMENDED_METRICS:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["recommendation_score"] = out.apply(_score_row, axis=1)
    out["execution_relevance_score"] = out.apply(_score_row_execution, axis=1)
    out["target_sort"] = out["prediction_column"].apply(
        lambda x: TARGET_FAMILY_ORDER.index(x) if x in TARGET_FAMILY_ORDER else 999
    )
    out = out.sort_values(["target_sort", "prediction_column", "recommendation_score"], ascending=[True, True, False]).reset_index(drop=True)
    return out


def _build_unified_wide(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pred_col, part in df.groupby("prediction_column", sort=True):
        row: dict[str, Any] = {"prediction_column": pred_col}
        truth = part["truth_column"].dropna().astype(str).unique().tolist()
        row["truth_column"] = truth[0] if truth else ""
        for _, r in part.iterrows():
            label = str(r["model_label"])
            row[f"{label}__score"] = _safe_float(r.get("recommendation_score"))
            for m in RECOMMENDED_METRICS:
                row[f"{label}__{m}"] = _safe_float(r.get(m))
        rows.append(row)
    return pd.DataFrame(rows)


def _recommend_per_target(df: pd.DataFrame) -> pd.DataFrame:
    best = (
        df.sort_values(["prediction_column", "recommendation_score"], ascending=[True, False])
        .groupby("prediction_column", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    cols = ["prediction_column", "truth_column", "model_label", "recommendation_score", *RECOMMENDED_METRICS, "run_dir"]
    return best[[c for c in cols if c in best.columns]]


def _recommend_per_target_execution(df: pd.DataFrame) -> pd.DataFrame:
    best = (
        df.sort_values(["prediction_column", "execution_relevance_score"], ascending=[True, False])
        .groupby("prediction_column", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    cols = ["prediction_column", "truth_column", "model_label", "execution_relevance_score", *RECOMMENDED_METRICS, "run_dir"]
    return best[[c for c in cols if c in best.columns]]


def _load_preflight(run_dir: Path, split: str) -> dict[str, Any]:
    p = run_dir / f"preflight_{split}" / "preflight_summary.json"
    if not p.exists():
        return {"exists": False, "path": str(p), "final_ok": None}
    payload = _read_json(p)
    payload["exists"] = True
    payload["path"] = str(p)
    return payload


def _load_checksums(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "training_run_context.json"
    if not p.exists():
        return {}
    payload = _read_json(p)
    return payload.get("input_checksums", {}) or {}


def _open_risks(df: pd.DataFrame, models: list[ModelInput], split: str) -> list[str]:
    risks: list[str] = []

    # Coverage parity.
    by_model = {m.label: set(df.loc[df["model_label"] == m.label, "prediction_column"].astype(str)) for m in models}
    if by_model:
        inter = set.intersection(*by_model.values())
        union = set.union(*by_model.values())
        if inter != union:
            risks.append(
                "Model target coverage is not identical across runs. "
                "Only compare common targets for final claims."
            )

    # Preflight state.
    for m in models:
        pf = _load_preflight(m.run_dir, split)
        if not pf.get("exists", False):
            risks.append(f"{m.label}: preflight report missing ({pf.get('path')}).")
            continue
        if not bool(pf.get("final_ok", False)):
            risks.append(
                f"{m.label}: preflight final_ok=False "
                "(schema/NaN/quantile/range issue unresolved)."
            )

    # Gate-metric availability.
    if "gate_mae" in df.columns:
        da_gate_missing = df.loc[df["prediction_column"] == "pred_da_price", "gate_mae"].isna().all()
        cap_gate_missing = (
            df.loc[df["prediction_column"].isin(["pred_afrr_capacity_price_pos", "pred_afrr_capacity_price_neg"]), "gate_mae"]
            .isna()
            .all()
        )
        if da_gate_missing:
            risks.append("DA gate-closure metrics are missing for all models.")
        if cap_gate_missing:
            risks.append("aFRR capacity gate-closure metrics are missing for all models.")

    # Quality flags (simple, explicit thresholds).
    weak_spike = df[(pd.to_numeric(df["spike_recall_top5_h1"], errors="coerce") < 0.20)]
    if not weak_spike.empty:
        touched = sorted(set(weak_spike["prediction_column"].astype(str).tolist()))
        risks.append(
            "Low spike recall (<0.20) on targets: " + ", ".join(touched) + ". "
            "High-volatility events are still weakly captured."
        )

    high_lag = df[pd.to_numeric(df["lag_best_h_h1"], errors="coerce").abs() > 6]
    if not high_lag.empty:
        touched = sorted(set(high_lag["prediction_column"].astype(str).tolist()))
        risks.append(
            "Large lag shift (|best lag| > 6h) on targets: "
            + ", ".join(touched)
            + ". Potential reaction delay risk in trading."
        )

    # Feature snapshot consistency check.
    da_hashes: set[str] = set()
    afrr_hashes: set[str] = set()
    for m in models:
        checks = _load_checksums(m.run_dir)
        da_sha = str((checks.get("da_base_dir", {}) or {}).get("feature_config_sha256", "")).strip()
        afrr_sha = str((checks.get("afrr_base_dir", {}) or {}).get("feature_config_sha256", "")).strip()
        if da_sha:
            da_hashes.add(da_sha)
        if afrr_sha:
            afrr_hashes.add(afrr_sha)
    if len(da_hashes) > 1:
        risks.append("DA feature-config checksum differs across compared runs.")
    if len(afrr_hashes) > 1:
        risks.append("aFRR feature-config checksum differs across compared runs.")

    if not risks:
        risks.append("No blocking open risks detected in current report package.")
    return risks


def _fmt_num(v: Any, digits: int = 4) -> str:
    x = _safe_float(v)
    if not np.isfinite(x):
        return "-"
    return f"{x:.{digits}f}"


def _markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, r in df.iterrows():
        row = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, (int, float, np.floating)):
                row.append(_fmt_num(v))
            else:
                row.append(str(v))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _write_markdown_report(
    out_path: Path,
    split: str,
    unified_df: pd.DataFrame,
    reco_df: pd.DataFrame,
    reco_exec_df: pd.DataFrame,
    risks: list[str],
    models: list[ModelInput],
) -> None:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Compact thesis-facing table.
    compact = unified_df.copy()
    compact = compact.sort_values("prediction_column")
    report_cols = [c for c in compact.columns if c.endswith("__score")]
    compact_cols = ["prediction_column", "truth_column", *sorted(report_cols)]
    compact = compact[compact_cols]

    reco_show = reco_df.copy()
    keep_reco_cols = [
        "prediction_column",
        "model_label",
        "recommendation_score",
        "mae_mean",
        "rmse_mean",
        "gate_mae",
        "directional_accuracy",
        "spike_recall_top5_h1",
    ]
    reco_show = reco_show[[c for c in keep_reco_cols if c in reco_show.columns]]
    reco_exec_show = reco_exec_df.copy()
    keep_exec_cols = [
        "prediction_column",
        "model_label",
        "execution_relevance_score",
        "gate_dplus1_mae",
        "gate_dplus1_rmse",
        "spread_directional_error",
        "gate_dplus1_directional_accuracy",
        "mae_mean",
    ]
    reco_exec_show = reco_exec_show[[c for c in keep_exec_cols if c in reco_exec_show.columns]]

    model_lines = "\n".join([f"- `{m.label}`: `{m.run_dir}`" for m in models])
    risk_lines = "\n".join([f"- {r}" for r in risks])

    md = f"""# Final Benchmark Report (Split: {split})

Generated: {generated}

## Compared Runs
{model_lines}

## 1) Unified Comparison Table (Test Split)
The table below compares all models on a unified, target-level score used for model selection before backtest.

{_markdown_table(compact, compact_cols)}

## 2) Recommendation Per Target
Highest-scoring model by target based on gate-aware error, directional quality, spike handling, and skill.

{_markdown_table(reco_show, [c for c in reco_show.columns])}

## 3) Execution-Relevant (Gate-Time) Performance
Highest-scoring model by target under strict gate-time D+1 criteria.

{_markdown_table(reco_exec_show, [c for c in reco_exec_show.columns])}

## 4) Explicit Open-Risk List Before Backtest
{risk_lines}

## Notes for Thesis Use
- Use this report for final model-selection transparency before simulation.
- Distinguish global winner from gate-time execution winner in final thesis text.
- Keep split discipline: `val` for tuning, `test` for final claims.
- Archive this report together with simulation run IDs used for final chapters.
"""
    out_path.write_text(md, encoding="utf-8")


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate final benchmark package for thesis reporting.")
    p.add_argument("--run-dirs", nargs="+", required=True, help="Evaluated run directories.")
    p.add_argument("--labels", nargs="+", required=True, help="Model labels (same order/length as run dirs).")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--out-dir", default="artifacts/benchmarks/final_report", help="Output directory.")
    p.add_argument(
        "--allow-mixed-run-dirs",
        action="store_true",
        help="Allow run dirs containing multiple model families (not recommended).",
    )
    return p


def main() -> None:
    args = _build_cli().parse_args()
    run_dirs = [Path(p) for p in args.run_dirs]
    labels = args.labels
    if len(run_dirs) != len(labels):
        raise ValueError("--run-dirs and --labels must have the same length.")

    models = [ModelInput(label=lbl, run_dir=rd) for lbl, rd in zip(labels, run_dirs)]
    for m in models:
        fam = _detect_run_families(m.run_dir)
        expected = _label_to_expected_family(m.label)
        if (not args.allow_mixed_run_dirs) and len(fam) > 1:
            raise ValueError(
                f"Run dir '{m.run_dir}' contains mixed model families {sorted(fam)}. "
                "Use one run dir per model for fair comparison."
            )
        if expected is not None and fam and expected not in fam:
            raise ValueError(
                f"Label '{m.label}' expects family '{expected}', but run dir '{m.run_dir}' "
                f"looks like {sorted(fam)}."
            )

    frames = [_load_rows(m, expected_split=args.split) for m in models]
    long_df = pd.concat(frames, axis=0, ignore_index=True)

    # Fairness rule: compare only common targets across all models.
    by_model = {
        m.label: set(long_df.loc[long_df["model_label"] == m.label, "prediction_column"].astype(str).tolist())
        for m in models
    }
    common_targets = set.intersection(*by_model.values()) if by_model else set()
    if not common_targets:
        raise ValueError("No common prediction targets across models; cannot benchmark fairly.")
    long_df = long_df[long_df["prediction_column"].isin(common_targets)].copy()

    norm_df = _normalize_table(long_df)
    unified_wide_df = _build_unified_wide(norm_df)
    reco_df = _recommend_per_target(norm_df)
    reco_exec_df = _recommend_per_target_execution(norm_df)
    risks = _open_risks(norm_df, models=models, split=args.split)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_csv = out_dir / f"unified_table_{args.split}_long.csv"
    wide_csv = out_dir / f"unified_table_{args.split}_wide.csv"
    reco_csv = out_dir / f"recommendation_per_target_{args.split}.csv"
    reco_exec_csv = out_dir / f"recommendation_per_target_execution_{args.split}.csv"
    risk_json = out_dir / f"open_risk_list_{args.split}.json"
    report_md = out_dir / f"final_benchmark_report_{args.split}.md"

    norm_df.to_csv(long_csv, index=False)
    unified_wide_df.to_csv(wide_csv, index=False)
    reco_df.to_csv(reco_csv, index=False)
    reco_exec_df.to_csv(reco_exec_csv, index=False)
    risk_json.write_text(json.dumps({"split": args.split, "open_risks": risks}, indent=2), encoding="utf-8")
    _write_markdown_report(
        out_path=report_md,
        split=args.split,
        unified_df=unified_wide_df,
        reco_df=reco_df,
        reco_exec_df=reco_exec_df,
        risks=risks,
        models=models,
    )

    print("[OK] Final benchmark report package generated.")
    print(f"- unified_long: {long_csv}")
    print(f"- unified_wide: {wide_csv}")
    print(f"- recommendations: {reco_csv}")
    print(f"- execution_recommendations: {reco_exec_csv}")
    print(f"- open_risks: {risk_json}")
    print(f"- report_md: {report_md}")


if __name__ == "__main__":
    main()
