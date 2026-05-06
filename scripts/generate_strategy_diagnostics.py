#!/usr/bin/env python3
"""Generate thesis-grade strategy diagnostics from simulation artifacts.

Covers:
1) Prediction-to-decision mapping (documented from implementation)
2) Strategy behavior decomposition (DA-only / aFRR-only / stacked)
3) Failure mode analysis (spike miss / lag / bias proxies)
4) Constraint interaction diagnostics (slack, missed capacity, SoC/final constraints)
5) Calibration-to-strategy link (quantile sweep + optional forecast calibration files)
6) Comparative strategic diagnosis across models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _infer_model_label(path: Path) -> str:
    s = str(path).lower()
    if "xgb" in s or "xgboost" in s:
        return "xgboost"
    if "linear" in s or "lear" in s:
        return "linear"
    if "tft" in s:
        return "tft"
    return "unknown"


def _safe_float(v: Any, default: float = np.nan) -> float:
    try:
        x = float(v)
        return x
    except Exception:
        return default


def _load_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _discover_scenarios(sim_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    # Preferred: quantile_sweep_summary.csv gives explicit scenario metadata.
    for sweep in sim_root.rglob("quantile_sweep_summary.csv"):
        try:
            sdf = pd.read_csv(sweep)
        except Exception:
            continue
        for r in sdf.to_dict(orient="records"):
            out_dir = Path(str(r.get("output_dir", "")))
            if not out_dir.is_absolute():
                out_dir = (sweep.parent / out_dir).resolve()
            summary_path = out_dir / "backtest_summary.json"
            hourly_path = out_dir / "backtest_hourly.parquet"
            if not summary_path.exists() or not hourly_path.exists():
                continue
            rows.append(
                {
                    "source": str(sweep),
                    "output_dir": str(out_dir),
                    "summary_path": str(summary_path),
                    "hourly_path": str(hourly_path),
                    "scenario": str(r.get("scenario", out_dir.name)),
                    "quantile_low": str(r.get("quantile_low", "")),
                    "quantile_high": str(r.get("quantile_high", "")),
                    "da_quantile_role": str(r.get("da_quantile_role", "mid")),
                    "model": _infer_model_label(out_dir),
                }
            )

    # Fallback: any direct backtest_summary/hourly pair.
    if not rows:
        for summary in sim_root.rglob("backtest_summary.json"):
            out_dir = summary.parent
            hourly = out_dir / "backtest_hourly.parquet"
            if not hourly.exists():
                continue
            rows.append(
                {
                    "source": "fallback_discovery",
                    "output_dir": str(out_dir),
                    "summary_path": str(summary),
                    "hourly_path": str(hourly),
                    "scenario": out_dir.name,
                    "quantile_low": "",
                    "quantile_high": "",
                    "da_quantile_role": "mid",
                    "model": _infer_model_label(out_dir),
                }
            )

    df = pd.DataFrame(rows).drop_duplicates(subset=["output_dir"]) if rows else pd.DataFrame()
    return df


def _compute_scenario_metrics(scenarios: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    for r in scenarios.to_dict(orient="records"):
        sp = Path(r["summary_path"])
        hp = Path(r["hourly_path"])
        summary = _load_json(sp)
        hourly = pd.read_parquet(hp)

        row = dict(r)
        row.update(
            {
                "realized_total_pnl_eur": _safe_float(summary.get("realized_total_pnl_eur")),
                "predicted_total_pnl_eur": _safe_float(summary.get("predicted_total_pnl_eur")),
                "naive_total_pnl_eur": _safe_float(summary.get("naive_total_pnl_eur")),
                "oracle_total_pnl_eur": _safe_float(summary.get("oracle_total_pnl_eur")),
                "stacking_value_realized_eur": _safe_float(summary.get("stacking_value_realized_eur")),
                "realized_da_only_total_pnl_eur": _safe_float(summary.get("realized_da_only_total_pnl_eur")),
                "realized_afrr_only_total_pnl_eur": _safe_float(summary.get("realized_afrr_only_total_pnl_eur")),
                "total_penalty_cost_eur": _safe_float(summary.get("total_penalty_cost_eur", 0.0), 0.0),
                "total_penalty_activation_eur": _safe_float(summary.get("total_penalty_activation_eur", 0.0), 0.0),
                "total_penalty_capacity_eur": _safe_float(summary.get("total_penalty_capacity_eur", 0.0), 0.0),
                "total_missed_activation_mwh": _safe_float(summary.get("total_missed_activation_mwh", 0.0), 0.0),
                "total_missed_capacity_mw": _safe_float(summary.get("total_missed_capacity_mw", 0.0), 0.0),
                "total_slack_pos_mw": _safe_float(summary.get("total_slack_pos_mw", 0.0), 0.0),
                "total_slack_neg_mw": _safe_float(summary.get("total_slack_neg_mw", 0.0), 0.0),
                "final_soc_constraint_satisfied": _safe_float(summary.get("final_soc_constraint_satisfied", np.nan)),
                "decision_volatility_total_flips": _safe_float(summary.get("decision_volatility_total_flips", 0.0), 0.0),
                "decision_volatility_mean_abs_revision_mw": _safe_float(summary.get("decision_volatility_mean_abs_revision_mw", 0.0), 0.0),
                "capture_ratio_vs_oracle": _safe_float(summary.get("capture_ratio_vs_oracle", np.nan)),
                "capture_ratio_vs_naive": _safe_float(summary.get("capture_ratio_vs_naive", np.nan)),
                "roi_on_max_capital": _safe_float(summary.get("roi_on_max_capital", np.nan)),
                "total_equivalent_full_cycles": _safe_float(summary.get("total_equivalent_full_cycles", np.nan)),
            }
        )

        # Failure-mode proxies from hourly table.
        pnl_col = "real_pnl_eur" if "real_pnl_eur" in hourly.columns else None
        if pnl_col:
            pnl = pd.to_numeric(hourly[pnl_col], errors="coerce").fillna(0.0)
            abs_total = float(pnl.abs().sum())
            top_k = max(1, int(np.ceil(0.01 * len(pnl))))
            row["pnl_top1pct_concentration_share"] = float(pnl.abs().nlargest(top_k).sum() / abs_total) if abs_total > 1e-12 else np.nan
        else:
            row["pnl_top1pct_concentration_share"] = np.nan

        pred_gap = row["predicted_total_pnl_eur"] - row["realized_total_pnl_eur"]
        row["failure_bias_pnl_gap_eur"] = pred_gap
        row["failure_spike_miss_ratio"] = (
            row["total_penalty_activation_eur"] / row["total_penalty_cost_eur"]
            if abs(row["total_penalty_cost_eur"]) > 1e-12 else np.nan
        )
        row["failure_lag_proxy"] = (
            row["decision_volatility_total_flips"] * row["decision_volatility_mean_abs_revision_mw"]
        )
        row["constraint_pressure_index"] = (
            row["total_slack_pos_mw"] + row["total_slack_neg_mw"] + row["total_missed_capacity_mw"]
        )

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def _build_prediction_to_decision_mapping_md(out_path: Path) -> None:
    txt = """# Prediction-to-Decision Mapping (Implementation)

This mapping reflects `src/energy_trading/simulation/battery_backtest.py`.

## MILP objective inputs (ex-ante)
- `pred_da_price` enters DA charge/discharge objective coefficients.
- `pred_cap_pos`, `pred_cap_neg` enter capacity revenue term.
- `pred_act_pos`, `pred_act_neg` enter expected activation revenue term.
- `pred_rate_pos`, `pred_rate_neg` (or quantile-selected rates) scale expected activation energy in chance constraints/objective.
- Soft-constraint slacks `slack_pos`, `slack_neg` are penalized by `imbalance_penalty_eur_mwh * dt_h`.

## Sign conventions
- Charging is a cost-side decision; discharging is revenue-side.
- Positive/negative aFRR are modeled with separate directional variables (`pos`, `neg`), then aggregated.
- Net PnL = revenues - costs - transaction - degradation - penalties.

## Quantiles vs point forecasts
- Simulation supports quantile-pair scenarios (`p10-p90`, etc.) via `--quantile-pairs`.
- These quantiles affect market inputs used for dispatch and therefore bidding aggressiveness.
- Default (no sweep) runs use baseline forecast columns.
"""
    out_path.write_text(txt, encoding="utf-8")


def _comparative_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    for model, g in df.groupby("model"):
        g2 = g.copy()
        # best scenario by realized pnl
        g2 = g2.sort_values("realized_total_pnl_eur", ascending=False)
        best = g2.iloc[0]
        rows.append(
            {
                "model": model,
                "best_scenario": best.get("scenario"),
                "best_quantile_pair": f"{best.get('quantile_low','')}-{best.get('quantile_high','')}",
                "best_realized_pnl_eur": _safe_float(best.get("realized_total_pnl_eur")),
                "mean_realized_pnl_eur": _safe_float(g["realized_total_pnl_eur"].mean()),
                "mean_capture_ratio_vs_oracle": _safe_float(g["capture_ratio_vs_oracle"].mean()),
                "mean_penalty_eur": _safe_float(g["total_penalty_cost_eur"].mean()),
                "mean_constraint_pressure_index": _safe_float(g["constraint_pressure_index"].mean()),
                "mean_spike_miss_ratio": _safe_float(g["failure_spike_miss_ratio"].mean()),
                "mean_bias_pnl_gap_eur": _safe_float(g["failure_bias_pnl_gap_eur"].mean()),
                "recommendation": (
                    "Use aggressive upper quantiles only with high penalty tolerance"
                    if _safe_float(g["mean_constraint_pressure_index"].mean(), 0.0) > 0.0
                    else "Use model as primary with standard uncertainty guardrails"
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate thesis strategy diagnostics from simulation outputs.")
    ap.add_argument("--simulation-root", default="artifacts/simulation_runs")
    ap.add_argument("--out-dir", default="artifacts/analysis/strategy_diagnostics")
    args = ap.parse_args()

    sim_root = Path(args.simulation_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _discover_scenarios(sim_root)
    if scenarios.empty:
        raise FileNotFoundError(f"No scenarios found under {sim_root}")

    scenarios.to_csv(out_dir / "discovered_scenarios.csv", index=False)
    metrics = _compute_scenario_metrics(scenarios)
    metrics.to_csv(out_dir / "scenario_diagnostics.csv", index=False)

    # Strategy behavior decomposition (you asked for this explicitly)
    decomp_cols = [
        "model",
        "scenario",
        "quantile_low",
        "quantile_high",
        "realized_da_only_total_pnl_eur",
        "realized_afrr_only_total_pnl_eur",
        "stacking_value_realized_eur",
        "realized_total_pnl_eur",
    ]
    decomp = metrics[[c for c in decomp_cols if c in metrics.columns]].copy()
    decomp.to_csv(out_dir / "strategy_behavior_decomposition.csv", index=False)

    # Constraint interaction report
    constraint_cols = [
        "model",
        "scenario",
        "total_slack_pos_mw",
        "total_slack_neg_mw",
        "total_missed_capacity_mw",
        "final_soc_constraint_satisfied",
        "constraint_pressure_index",
    ]
    metrics[[c for c in constraint_cols if c in metrics.columns]].to_csv(
        out_dir / "constraint_interaction_report.csv", index=False
    )

    # Failure modes report
    failure_cols = [
        "model",
        "scenario",
        "failure_spike_miss_ratio",
        "failure_lag_proxy",
        "failure_bias_pnl_gap_eur",
        "total_penalty_activation_eur",
        "total_penalty_cost_eur",
        "pnl_top1pct_concentration_share",
    ]
    metrics[[c for c in failure_cols if c in metrics.columns]].to_csv(
        out_dir / "failure_mode_report.csv", index=False
    )

    # Comparative diagnosis across models
    comp = _comparative_diagnosis(metrics)
    comp.to_csv(out_dir / "comparative_strategic_diagnosis.csv", index=False)

    # Calibration-to-strategy link (quantile pair vs realized pnl)
    cal_link = metrics[[c for c in [
        "model", "scenario", "quantile_low", "quantile_high", "realized_total_pnl_eur",
        "capture_ratio_vs_oracle", "total_penalty_cost_eur", "constraint_pressure_index"
    ] if c in metrics.columns]].copy()
    cal_link.to_csv(out_dir / "calibration_to_strategy_link.csv", index=False)

    # Human-readable mapping doc.
    _build_prediction_to_decision_mapping_md(out_dir / "prediction_to_decision_mapping.md")

    summary = {
        "n_scenarios": int(len(metrics)),
        "models": sorted(set(metrics.get("model", pd.Series(dtype=str)).dropna().astype(str).tolist())),
        "outputs": [
            "discovered_scenarios.csv",
            "scenario_diagnostics.csv",
            "strategy_behavior_decomposition.csv",
            "constraint_interaction_report.csv",
            "failure_mode_report.csv",
            "calibration_to_strategy_link.csv",
            "comparative_strategic_diagnosis.csv",
            "prediction_to_decision_mapping.md",
        ],
    }
    (out_dir / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
