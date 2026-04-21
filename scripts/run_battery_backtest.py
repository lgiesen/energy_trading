"""Run LP-based battery backtest from ML predictions + ground truth parquet files.

Usage (manifest-autoload, recommended):
    ./.venv/bin/python scripts/run_battery_backtest.py \
      --run-manifest artifacts/model_runs/latest.json \
      --split test \
      --horizon-hours 48 \
      --reopt-step-hours 1 \
      --da-gate-hour-utc 11 \
      --soc-feedback-mode realized

Usage (manual files):
    ./.venv/bin/python scripts/run_battery_backtest.py \
      --predictions artifacts/simulation_runs/manual/backtest_table_test.parquet \
      --ground-truth data/features/all_data_features.parquet \
      --timestamp-col timestamp_utc \
      --pred-da-col pred_da_price \
      --true-da-col da_price
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow direct script execution from repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_trading.simulation.battery_backtest import (
    BacktestColumnMap,
    BatteryBacktester,
    load_and_align_market_data,
    load_prediction_warehouse_long,
)


def _plot_cumulative_pnl(hourly: pd.DataFrame, ts_col: str, out_path: Path) -> None:
    if hourly.empty:
        return
    d = hourly.copy()
    d[ts_col] = pd.to_datetime(d[ts_col], utc=True, errors="coerce")
    d = d.dropna(subset=[ts_col]).sort_values(ts_col)
    required = ["real_pnl_eur", "naive_pnl_eur", "oracle_pnl_eur"]
    if not set(required).issubset(d.columns):
        return
    d["model_cum_pnl_eur"] = pd.to_numeric(d["real_pnl_eur"], errors="coerce").fillna(0.0).cumsum()
    d["naive_cum_pnl_eur"] = pd.to_numeric(d["naive_pnl_eur"], errors="coerce").fillna(0.0).cumsum()
    d["oracle_cum_pnl_eur"] = pd.to_numeric(d["oracle_pnl_eur"], errors="coerce").fillna(0.0).cumsum()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(d[ts_col], d["model_cum_pnl_eur"], label="Model", linewidth=2)
    ax.plot(d[ts_col], d["naive_cum_pnl_eur"], label="Naive 24h", linewidth=2)
    ax.plot(d[ts_col], d["oracle_cum_pnl_eur"], label="Oracle", linewidth=2)
    ax.set_title("Cumulative PnL Contribution")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Cumulative PnL [EUR]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _build_backtest_diagnostics(hourly: pd.DataFrame, summary: dict[str, object]) -> dict[str, object]:
    d = hourly.copy()
    numeric = d.select_dtypes(include=["number"])
    nonfinite_total = int((~np.isfinite(numeric.to_numpy(dtype=float))).sum()) if not numeric.empty else 0
    nan_total = int(numeric.isna().sum().sum()) if not numeric.empty else 0

    key_cols = [
        "real_pnl_eur",
        "pred_pnl_eur",
        "naive_pnl_eur",
        "oracle_pnl_eur",
        "soc_mwh",
        "charge_mw",
        "discharge_mw",
        "reserve_pos_mw",
        "reserve_neg_mw",
    ]
    key_col_nan_counts = {
        c: int(pd.to_numeric(d[c], errors="coerce").isna().sum())
        for c in key_cols
        if c in d.columns
    }

    infeasibility_flags = {
        "final_soc_constraint_satisfied": bool(summary.get("final_soc_constraint_satisfied", False)),
        "numeric_nonfinite_total": nonfinite_total,
    }
    return {
        "rows_hourly": int(len(d)),
        "numeric_nan_total": nan_total,
        "numeric_nonfinite_total": nonfinite_total,
        "key_column_nan_counts": key_col_nan_counts,
        "infeasibility_flags": infeasibility_flags,
    }


def _resolve_out_dir(
    out_dir_arg: str,
    *,
    run_id: str | None,
    split: str,
) -> Path:
    if out_dir_arg.strip():
        out = Path(out_dir_arg)
        out.mkdir(parents=True, exist_ok=True)
        return out
    rid = run_id or "manual"
    out = Path("artifacts/simulation_runs") / rid / split
    out.mkdir(parents=True, exist_ok=True)
    return out


def _matches_model_key(path: Path, model_key: str) -> bool:
    if not model_key:
        return True
    mk = model_key.strip().lower()
    name = path.name.lower()
    if mk in {"xgb", "xgboost"}:
        return "xgboost" in name
    if mk == "tft":
        return "xgboost" not in name
    return mk in name


def _resolve_existing_file(path_like: str | Path, *, manifest_dir: Path) -> Path:
    p = Path(path_like)
    if p.exists():
        return p
    repo_root = Path(__file__).resolve().parents[1]
    # Common case for downloaded artifacts: manifest keeps server-absolute path.
    cands = [
        manifest_dir / p.name,
        manifest_dir / "predictions" / p.name,
        Path.cwd() / p.name,
        Path.cwd() / "data" / "features" / p.name,
        repo_root / "data" / "features" / p.name,
        repo_root / "data" / "features" / "all_data_features.parquet",
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"File not found from manifest path '{p}'. Tried local fallbacks near {manifest_dir}.")


def _resolve_long_prediction_path(
    *,
    pred_col: str,
    configured_path: str | Path,
    manifest_dir: Path,
    split: str,
    model_key: str,
) -> Path:
    p = Path(configured_path)
    if p.exists() and _matches_model_key(p, model_key):
        return p

    pred_dir = manifest_dir / "predictions"
    candidates: list[Path] = []
    if pred_dir.exists():
        patterns = [
            f"*{split}*{pred_col}*long*.parquet",
            f"*{pred_col}*long*{split}*.parquet",
            f"*{pred_col}*{split}*long*.parquet",
        ]
        for pat in patterns:
            candidates.extend(sorted(pred_dir.glob(pat)))
    if model_key:
        candidates = [c for c in candidates if _matches_model_key(c, model_key)]
    if candidates:
        return candidates[0]

    # Final explicit fallbacks by filename
    for c in [manifest_dir / p.name, manifest_dir / "predictions" / p.name]:
        if c.exists() and _matches_model_key(c, model_key):
            return c
    raise FileNotFoundError(
        f"Could not resolve long prediction file for pred_col='{pred_col}', split='{split}', model_key='{model_key}'. "
        f"Configured path: {p}"
    )


def _resolve_long_map(
    *,
    long_map: dict[str, str],
    manifest_dir: Path,
    split: str,
    model_key: str,
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for pred_col, p in long_map.items():
        rp = _resolve_long_prediction_path(
            pred_col=pred_col,
            configured_path=p,
            manifest_dir=manifest_dir,
            split=split,
            model_key=model_key,
        )
        resolved[pred_col] = str(rp)
    return resolved


def _apply_fallback_column_map(pred: pd.DataFrame, truth: pd.DataFrame, colmap: BacktestColumnMap) -> BacktestColumnMap:
    """Use project-aware fallback candidates to reduce manual mapping overhead."""

    def pick(frame: pd.DataFrame, primary: str, candidates: list[str]) -> str:
        for c in [primary, *candidates]:
            if c in frame.columns:
                return c
        raise KeyError(f"Missing required column. Tried: {[primary, *candidates]}")

    def pick_pred(frame: pd.DataFrame, primary: str, candidates: list[str]) -> str:
        # In long-warehouse mode pred preview can be empty. Keep configured names.
        if frame.empty:
            return primary
        return pick(frame, primary, candidates)

    mapped = BacktestColumnMap(
        timestamp=pick(pred if colmap.timestamp in pred.columns else truth, colmap.timestamp, ["timestamp", "datetime", "date"]),

        pred_da_price=pick_pred(
            pred,
            colmap.pred_da_price,
            ["da_price_pred", "y_pred_da_price", "prediction_da_price", "pred_target_da_price"],
        ),
        pred_afrr_capacity_price_pos=pick_pred(pred, colmap.pred_afrr_capacity_price_pos, ["afrr_capacity_price_pos_pred", "pred_target_afrr_capacity_price_pos"]),
        pred_afrr_capacity_price_neg=pick_pred(pred, colmap.pred_afrr_capacity_price_neg, ["afrr_capacity_price_neg_pred", "pred_target_afrr_capacity_price_neg"]),
        pred_afrr_activation_price_pos=pick_pred(pred, colmap.pred_afrr_activation_price_pos, ["afrr_activation_price_vwap_pos_pred", "pred_target_afrr_activation_price_vwap_pos"]),
        pred_afrr_activation_price_neg=pick_pred(
            pred,
            colmap.pred_afrr_activation_price_neg,
            ["afrr_activation_price_vwap_neg_pred", "pred_target_afrr_activation_price_vwap_neg"],
        ),
        pred_afrr_activation_rate_pos=pick_pred(
            pred,
            colmap.pred_afrr_activation_rate_pos,
            ["pred_target_afrr_activation_rate_pos", "afrr_activation_rate_pred"],
        ),
        pred_afrr_activation_rate_neg=pick_pred(
            pred,
            colmap.pred_afrr_activation_rate_neg,
            ["pred_target_afrr_activation_rate_neg", "afrr_activation_rate_pred"],
        ),

        true_da_price=pick(truth, colmap.true_da_price, ["da_price_actual", "target_da_price"]),
        true_afrr_capacity_price_pos=pick(truth, colmap.true_afrr_capacity_price_pos, ["target_afrr_capacity_price_pos"]),
        true_afrr_capacity_price_neg=pick(truth, colmap.true_afrr_capacity_price_neg, ["target_afrr_capacity_price_neg"]),
        true_afrr_activation_price_pos=pick(
            truth,
            colmap.true_afrr_activation_price_pos,
            ["target_afrr_activation_price_vwap_pos", "afrr_activation_price_vwap"],
        ),
        true_afrr_activation_price_neg=pick(
            truth,
            colmap.true_afrr_activation_price_neg,
            [
                "target_afrr_activation_price_vwap_neg",
                "afrr_activation_price_vwap",
                "afrr_activation_price_vwap_pos",
                "target_afrr_activation_price_vwap_pos",
            ],
        ),
        true_afrr_activation_rate_pos=pick(
            truth,
            colmap.true_afrr_activation_rate_pos,
            [
                "activation_rate_phys_pos",
                "afrr_activation_rate_pos",
                "target_afrr_activation_rate_pos",
                "afrr_activation_rate",
            ],
        ),
        true_afrr_activation_rate_neg=pick(
            truth,
            colmap.true_afrr_activation_rate_neg,
            [
                "activation_rate_phys_neg",
                "afrr_activation_rate_neg",
                "target_afrr_activation_rate_neg",
                "afrr_activation_rate",
            ],
        ),
    )
    if mapped.true_afrr_activation_price_neg == mapped.true_afrr_activation_price_pos:
        print(
            "[WARN] Missing dedicated negative activation-price truth column. "
            "Using positive activation-price column as fallback for *_neg settlement."
        )
    return mapped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run battery LP backtest with predicted-vs-realized settlement.")
    p.add_argument("--predictions", default="", help="Path to predictions parquet file.")
    p.add_argument("--ground-truth", default="", help="Path to ground-truth parquet file.")
    p.add_argument(
        "--run-manifest",
        default="artifacts/model_runs/latest.json",
        help="Manifest path or latest-pointer json for simulation autoload.",
    )
    p.add_argument("--split", choices=["val", "test"], default="test", help="Prediction split for manifest mode.")
    p.add_argument(
        "--model-key",
        default="",
        help="Optional model selector when one run dir contains multiple models (e.g. 'xgboost' or 'tft').",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory for hourly/aggregated results. If empty, uses artifacts/simulation_runs/<run_id>/<split>/",
    )

    p.add_argument("--timestamp-col", default="timestamp_utc")
    p.add_argument("--pred-da-col", default="pred_da_price")
    p.add_argument("--pred-cap-pos-col", default="pred_afrr_capacity_price_pos")
    p.add_argument("--pred-cap-neg-col", default="pred_afrr_capacity_price_neg")
    p.add_argument("--pred-act-pos-col", default="pred_afrr_activation_price_pos")
    p.add_argument("--pred-act-neg-col", default="pred_afrr_activation_price_neg")
    p.add_argument("--pred-rate-pos-col", default="pred_afrr_activation_rate_pos")
    p.add_argument("--pred-rate-neg-col", default="pred_afrr_activation_rate_neg")

    p.add_argument("--true-da-col", default="da_price")
    p.add_argument("--true-cap-pos-col", default="afrr_capacity_price_pos")
    p.add_argument("--true-cap-neg-col", default="afrr_capacity_price_neg")
    p.add_argument("--true-act-pos-col", default="afrr_activation_price_vwap_pos")
    p.add_argument("--true-act-neg-col", default="afrr_activation_price_vwap_neg")
    p.add_argument("--true-rate-pos-col", default="activation_rate_phys_pos")
    p.add_argument("--true-rate-neg-col", default="activation_rate_phys_neg")

    p.add_argument("--start", default=None, help="Optional UTC start filter.")
    p.add_argument("--end", default=None, help="Optional UTC end filter.")
    p.add_argument("--horizon-hours", type=int, default=48, help="Rolling-horizon window length in hours.")
    p.add_argument("--reopt-step-hours", type=int, default=1, help="Re-optimization step in hours.")
    p.add_argument("--da-gate-hour-utc", type=int, default=11, help="UTC gate-closure hour for locking next-day DA bids.")
    p.add_argument(
        "--soc-feedback-mode",
        choices=["realized", "predicted"],
        default="realized",
        help="State carryover mode between re-optimizations.",
    )
    p.add_argument(
        "--enforce-final-soc-min",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce final SoC >= soc_target_end (default: enabled).",
    )
    p.add_argument(
        "--disable-rolling-horizon",
        action="store_true",
        help="Use a single full-horizon optimization instead of rolling horizon.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_id: str | None = None

    predictions_path = args.predictions.strip()
    ground_truth_path = args.ground_truth.strip()

    if not predictions_path:
        manifest_path = Path(args.run_manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Run manifest/latest pointer not found: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "manifest_path" in payload:
            run_id = payload.get("run_id")
            manifest_path = Path(payload["manifest_path"])
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = run_id or payload.get("run_id")
        manifest_dir = manifest_path.parent
    else:
        manifest_dir = Path.cwd()

    out_dir = _resolve_out_dir(args.out_dir, run_id=run_id, split=args.split)
    forecast_warehouse: dict[str, pd.DataFrame] | None = None
    coverage_min: pd.Timestamp | None = None
    coverage_max: pd.Timestamp | None = None

    if not predictions_path:
        da_long = payload.get("bundles", {}).get("da", {}).get("predictions_long", {}).get(args.split, {})
        afrr_long = payload.get("bundles", {}).get("afrr", {}).get("predictions_long", {}).get(args.split, {})
        long_map = {**da_long, **afrr_long}

        if long_map:
            resolved_long_map = _resolve_long_map(
                long_map=long_map,
                manifest_dir=manifest_dir,
                split=args.split,
                model_key=args.model_key.strip(),
            )
            forecast_warehouse = load_prediction_warehouse_long(resolved_long_map)
            print(f"[INFO] Long-format forecast warehouse loaded for split='{args.split}' with {len(long_map)} files.")
            cov_min_list: list[pd.Timestamp] = []
            cov_max_list: list[pd.Timestamp] = []
            for wdf in forecast_warehouse.values():
                t = pd.to_datetime(wdf["target_time_utc"], utc=True, errors="coerce").dropna()
                if not t.empty:
                    cov_min_list.append(t.min())
                    cov_max_list.append(t.max())
            if cov_min_list and cov_max_list:
                coverage_min = min(cov_min_list)
                coverage_max = max(cov_max_list)
        else:
            da_pred = _resolve_existing_file(payload["bundles"]["da"]["predictions"][args.split], manifest_dir=manifest_dir)
            afrr_pred = _resolve_existing_file(payload["bundles"]["afrr"]["predictions"][args.split], manifest_dir=manifest_dir)
            predictions_path = str((out_dir / f"backtest_table_{args.split}.parquet").resolve())

            da_df = pd.read_parquet(da_pred)
            afrr_df = pd.read_parquet(afrr_pred)
            backtest_table = da_df.merge(afrr_df, on="timestamp_utc", how="inner")
            backtest_table.to_parquet(predictions_path, index=False)
            print(f"[INFO] Backtest table created: {predictions_path}")

        if not ground_truth_path:
            ground_truth_path = str(
                _resolve_existing_file(payload["ground_truth"]["default_path"], manifest_dir=manifest_dir)
            )

    if not ground_truth_path:
        raise ValueError("Either provide --predictions and --ground-truth, or use --run-manifest.")
    truth_preview = pd.read_parquet(ground_truth_path)
    pred_preview = pd.read_parquet(predictions_path) if predictions_path else pd.DataFrame()

    colmap_in = BacktestColumnMap(
        timestamp=args.timestamp_col,
        pred_da_price=args.pred_da_col,
        pred_afrr_capacity_price_pos=args.pred_cap_pos_col,
        pred_afrr_capacity_price_neg=args.pred_cap_neg_col,
        pred_afrr_activation_price_pos=args.pred_act_pos_col,
        pred_afrr_activation_price_neg=args.pred_act_neg_col,
        pred_afrr_activation_rate_pos=args.pred_rate_pos_col,
        pred_afrr_activation_rate_neg=args.pred_rate_neg_col,
        true_da_price=args.true_da_col,
        true_afrr_capacity_price_pos=args.true_cap_pos_col,
        true_afrr_capacity_price_neg=args.true_cap_neg_col,
        true_afrr_activation_price_pos=args.true_act_pos_col,
        true_afrr_activation_price_neg=args.true_act_neg_col,
        true_afrr_activation_rate_pos=args.true_rate_pos_col,
        true_afrr_activation_rate_neg=args.true_rate_neg_col,
    )

    # Always apply fallback mapping, even in long-warehouse mode (pred_preview can
    # be empty there but truth-side fallbacks are still useful).
    colmap = _apply_fallback_column_map(pred_preview, truth_preview, colmap_in)

    if predictions_path:
        df = load_and_align_market_data(predictions_path, ground_truth_path, colmap)
    else:
        df = truth_preview.copy()
        if colmap.timestamp not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df[colmap.timestamp] = df.index
        df[colmap.timestamp] = pd.to_datetime(df[colmap.timestamp], utc=True, errors="coerce")
        df = df.dropna(subset=[colmap.timestamp]).sort_values(colmap.timestamp).reset_index(drop=True)
        # In long-warehouse mode, keep only rows that are covered by forecast targets.
        if forecast_warehouse and coverage_min is not None and coverage_max is not None:
            df = df[(df[colmap.timestamp] >= coverage_min) & (df[colmap.timestamp] <= coverage_max)].copy()
    if args.start:
        start = pd.to_datetime(args.start, utc=True)
        df = df[df[colmap.timestamp] >= start].copy()
    if args.end:
        end = pd.to_datetime(args.end, utc=True)
        df = df[df[colmap.timestamp] <= end].copy()
    if df.empty:
        raise ValueError("No rows after timestamp filtering.")

    backtester = BatteryBacktester()
    outputs = backtester.run(
        df,
        colmap,
        use_rolling_horizon=not args.disable_rolling_horizon,
        horizon_hours=args.horizon_hours,
        reopt_step_hours=args.reopt_step_hours,
        forecast_warehouse=forecast_warehouse,
        da_gate_hour_utc=args.da_gate_hour_utc,
        soc_feedback_mode=args.soc_feedback_mode,
        enforce_final_soc_min=args.enforce_final_soc_min,
    )

    hourly_path = out_dir / "backtest_hourly.parquet"
    plan_history_path = out_dir / "backtest_plan_history.parquet"
    global_plan_history_path = Path("artifacts/backtest_plan_history.parquet")
    global_plan_history_path.parent.mkdir(parents=True, exist_ok=True)
    volatility_path = out_dir / "backtest_decision_volatility.csv"
    monthly_path = out_dir / "backtest_monthly.csv"
    yearly_path = out_dir / "backtest_yearly.csv"
    summary_path = out_dir / "backtest_summary.json"
    diagnostics_path = out_dir / "backtest_diagnostics.json"
    diagnostics_txt_path = out_dir / "backtest_diagnostics.txt"
    pnl_plot_path = out_dir / "backtest_cumulative_pnl.png"

    outputs.hourly.to_parquet(hourly_path, index=False)
    outputs.plan_history.to_parquet(plan_history_path, index=False)
    outputs.plan_history.to_parquet(global_plan_history_path, index=False)
    outputs.volatility.to_csv(volatility_path, index=False)
    outputs.monthly.to_csv(monthly_path, index=False)
    outputs.yearly.to_csv(yearly_path, index=False)
    summary_path.write_text(json.dumps(outputs.summary, indent=2), encoding="utf-8")
    diagnostics = _build_backtest_diagnostics(outputs.hourly, outputs.summary)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    diagnostics_txt_path.write_text(
        "\n".join(
            [
                "Backtest Diagnostics",
                f"rows_hourly={diagnostics['rows_hourly']}",
                f"numeric_nan_total={diagnostics['numeric_nan_total']}",
                f"numeric_nonfinite_total={diagnostics['numeric_nonfinite_total']}",
                (
                    "final_soc_constraint_satisfied="
                    f"{diagnostics['infeasibility_flags']['final_soc_constraint_satisfied']}"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _plot_cumulative_pnl(outputs.hourly, colmap.timestamp, pnl_plot_path)

    print("[OK] Battery backtest completed.")
    print(f"- rows: {len(outputs.hourly)}")
    print(f"- realized_total_pnl_eur: {outputs.summary['realized_total_pnl_eur']:.2f}")
    print(f"- oracle_total_pnl_eur: {outputs.summary['oracle_total_pnl_eur']:.2f}")
    print(f"- predicted_total_pnl_eur: {outputs.summary['predicted_total_pnl_eur']:.2f}")
    print(f"- naive_total_pnl_eur: {outputs.summary['naive_total_pnl_eur']:.2f}")
    print(f"- cost_of_forecast_error_total_eur: {outputs.summary['cost_of_forecast_error_total_eur']:.2f}")
    print(f"- pnl_gap_total_eur: {outputs.summary['pnl_gap_total_eur']:.2f}")
    print(f"- economic_opportunity_gap_ratio: {outputs.summary['economic_opportunity_gap_ratio']:.4f}")
    print(f"- max_capital_required_eur: {outputs.summary['max_capital_required_eur']:.2f}")
    print(
        "- final_soc: "
        f"{outputs.summary['final_real_soc_mwh']:.2f} MWh "
        f"(min target {outputs.summary['final_soc_min_target_mwh']:.2f}, "
        f"ok={bool(outputs.summary['final_soc_constraint_satisfied'])})"
    )
    print(f"- hourly: {hourly_path}")
    print(f"- plan_history: {plan_history_path}")
    print(f"- plan_history_global: {global_plan_history_path}")
    print(f"- decision_volatility: {volatility_path}")
    print(f"- monthly: {monthly_path}")
    print(f"- yearly: {yearly_path}")
    print(f"- summary: {summary_path}")
    print(f"- diagnostics: {diagnostics_path}")
    print(f"- diagnostics_txt: {diagnostics_txt_path}")
    print(f"- pnl_contribution_plot: {pnl_plot_path}")


if __name__ == "__main__":
    main()
