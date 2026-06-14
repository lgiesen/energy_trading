#!/usr/bin/env python3
"""Read-only audit for the DA precommit decision pipeline.

The script inspects existing simulation artifacts and classifies where DA
candidate volume is lost: raw optimizer, sizing, postlock guard, final
lockbook replay, or settlement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linprog
except Exception:  # pragma: no cover - optional diagnostic dependency
    linprog = None


TOL = 1e-9


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported table format: {path}")


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)


def _first_text(group: pd.DataFrame, cols: Iterable[str], default: str = "") -> str:
    for col in cols:
        if col not in group.columns:
            continue
        vals = group[col].dropna().astype(str)
        vals = vals[~vals.str.strip().isin(["", "nan", "None"])]
        if not vals.empty:
            return str(vals.iloc[0])
    return default


def _first_num(group: pd.DataFrame, cols: Iterable[str]) -> float:
    for col in cols:
        if col not in group.columns:
            continue
        vals = pd.to_numeric(group[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not vals.empty:
            return float(vals.iloc[0])
    return float("nan")


def _classify_gate(row: pd.Series) -> tuple[str, str]:
    raw = float(row["raw_buy_mwh"] + row["raw_sell_mwh"])
    sized = float(row["sized_buy_mwh"] + row["sized_sell_mwh"])
    postlock = float(row["postlock_buy_mwh"] + row["postlock_sell_mwh"])
    selected = float(row["selected_buy_mwh"] + row["selected_sell_mwh"])
    accepted = float(row["accepted_buy_mwh"] + row["accepted_sell_mwh"])
    reason = str(row.get("zero_reason", "") or row.get("volume_loss_reason", ""))
    if raw <= TOL:
        return "economic_no_trade", reason or "no_raw_candidate"
    if sized <= TOL:
        return "sizer_zeroed", reason or str(row.get("bid_sizer_status", "sized_candidate_zeroed"))
    if postlock <= TOL:
        if "terminal" in reason:
            return "terminal_guard_zeroed", reason
        if "reconciliation" in reason:
            return "settlement_equiv_replay_zeroed", reason
        return "postlock_zeroed", reason or "postlock_candidate_zeroed"
    if selected <= TOL:
        if "terminal" in reason:
            return "terminal_guard_zeroed", reason
        return "postlock_zeroed", reason or "selected_lockable_zeroed"
    if accepted <= TOL:
        return "settlement_equiv_replay_zeroed", reason or "accepted_lockbook_zeroed"
    if accepted + TOL < sized:
        return "postlock_derated", reason or "candidate_derated_before_lockbook"
    pnl_delta = float(row.get("candidate_minus_incumbent_pnl_eur", np.nan))
    if np.isfinite(pnl_delta) and pnl_delta <= TOL:
        return "economic_no_trade", "candidate_not_above_incumbent"
    return "selected", "selected_nonzero"


def _discover_scenarios(root: Path) -> list[Path]:
    if (root / "backtest_summary.json").exists():
        return [root]
    scenarios = sorted(p.parent for p in root.rglob("backtest_summary.json"))
    return scenarios


def _path_da_totals(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    candidates = {
        "submitted_buy": [f"{prefix}_submitted_da_buy_mw", f"{prefix}_da_submitted_buy_mwh"],
        "submitted_sell": [f"{prefix}_submitted_da_sell_mw", f"{prefix}_da_submitted_sell_mwh"],
        "accepted_buy": [f"{prefix}_da_auction_accepted_buy_mwh", f"{prefix}_da_buy_mwh"],
        "accepted_sell": [f"{prefix}_da_auction_accepted_sell_mwh", f"{prefix}_da_sell_mwh"],
        "realized_buy": [f"{prefix}_da_buy_mwh", f"{prefix}_realized_da_buy_mwh"],
        "realized_sell": [f"{prefix}_da_sell_mwh", f"{prefix}_realized_da_sell_mwh"],
    }
    out: dict[str, float] = {}
    for name, cols in candidates.items():
        for col in cols:
            if col in df.columns:
                out[name] = float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
                break
        else:
            out[name] = 0.0
    return out


def _summary_float(summary: dict[str, object], names: Iterable[str], default: float) -> float:
    for name in names:
        try:
            val = float(summary.get(name, np.nan))
        except Exception:
            continue
        if np.isfinite(val):
            return float(val)
    return float(default)


def _sum_first_existing(df: pd.DataFrame, cols: Iterable[str]) -> float:
    for col in cols:
        if col in df.columns:
            return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
    return 0.0


def _pnl_components(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    if df.empty:
        return {}
    if prefix == "real":
        pnl_cols = ["real_pnl_eur", "real_total_pnl_eur", "realized_total_pnl_eur"]
        da_pnl_cols = ["real_da_pnl_eur"]
        da_revenue_cols = ["real_revenue_da_eur"]
        da_cost_cols = ["real_cost_da_eur"]
        id_pnl_cols = ["real_pnl_id_eur", "real_id_pnl_eur"]
    elif prefix == "naive":
        pnl_cols = ["naive_pnl_eur", "naive_total_pnl_eur"]
        da_pnl_cols = ["naive_da_pnl_eur"]
        da_revenue_cols = ["naive_revenue_da_eur"]
        da_cost_cols = ["naive_cost_da_eur"]
        id_pnl_cols = ["naive_pnl_id_eur", "naive_id_pnl_eur"]
    else:
        pnl_cols = ["perfect_foresight_pnl_eur", "rolling_pf_reported_total_pnl_eur"]
        da_pnl_cols = ["perfect_foresight_da_pnl_eur", "rolling_pf_da_pnl_eur"]
        da_revenue_cols = ["perfect_foresight_revenue_da_eur"]
        da_cost_cols = ["perfect_foresight_cost_da_eur"]
        id_pnl_cols = ["perfect_foresight_pnl_id_eur", "rolling_pf_id_pnl_eur"]
    da_pnl = _sum_first_existing(df, da_pnl_cols)
    da_revenue = _sum_first_existing(df, da_revenue_cols)
    da_cost = _sum_first_existing(df, da_cost_cols)
    if abs(float(da_pnl)) <= TOL and (abs(float(da_revenue)) > TOL or abs(float(da_cost)) > TOL):
        da_pnl = float(da_revenue) - float(da_cost)
    return {
        "pnl_eur": _sum_first_existing(df, pnl_cols),
        "da_pnl_eur": float(da_pnl),
        "da_revenue_eur": float(da_revenue),
        "da_cost_eur": float(da_cost),
        "id_pnl_eur": _sum_first_existing(df, id_pnl_cols),
        "degradation_cost_eur": _sum_first_existing(df, [f"{prefix}_degradation_cost_eur"]),
        "transaction_cost_eur": _sum_first_existing(df, [f"{prefix}_transaction_cost_eur"]),
        "aux_cost_eur": _sum_first_existing(df, [f"{prefix}_aux_cost_eur"]),
    }


def _forecast_quality(hourly: pd.DataFrame) -> dict[str, float | str]:
    if hourly.empty:
        return {"status": "unavailable_empty_hourly"}
    pred_col = "pred_da_price" if "pred_da_price" in hourly.columns else ""
    true_col = ""
    for candidate in ("target_da_price", "true_da_price", "da_price"):
        if candidate in hourly.columns:
            true_col = candidate
            break
    if not pred_col or not true_col:
        return {"status": "unavailable_missing_columns"}
    pred = pd.to_numeric(hourly[pred_col], errors="coerce")
    true = pd.to_numeric(hourly[true_col], errors="coerce")
    valid = pred.notna() & true.notna()
    if not bool(valid.any()):
        return {"status": "unavailable_nonfinite"}
    err = pred[valid].astype(float) - true[valid].astype(float)
    corr = pred[valid].corr(true[valid]) if int(valid.sum()) > 1 else np.nan
    out: dict[str, float | str] = {
        "status": "ok",
        "rows": float(valid.sum()),
        "mae_eur_mwh": float(err.abs().mean()),
        "bias_eur_mwh": float(err.mean()),
        "corr": float(corr) if np.isfinite(corr) else float("nan"),
        "true_col": true_col,
    }
    lag_col = "pred_da_price_naive_source_lag_hours"
    fallback_col = "pred_da_price_naive_fallback_used"
    mode_col = "pred_da_price_naive_source_mode"
    if lag_col in hourly.columns:
        lags = pd.to_numeric(hourly[lag_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not lags.empty:
            out["naive_source_min_lag_hours"] = float(lags.min())
            out["naive_source_median_lag_hours"] = float(lags.median())
    if fallback_col in hourly.columns:
        out["naive_source_fallback_rows"] = float(
            pd.to_numeric(hourly[fallback_col], errors="coerce").fillna(0.0).sum()
        )
    if mode_col in hourly.columns:
        modes = hourly[mode_col].dropna().astype(str)
        if not modes.empty:
            out["naive_source_modes"] = ",".join(sorted(set(modes.tolist()))[:5])
    return out


def _diagnostic_global_da_oracle(hourly: pd.DataFrame, summary: dict[str, object]) -> dict[str, float | str]:
    """Simple full-horizon DA-only oracle using true prices.

    This is a diagnostic upper-audit, not simulation settlement. It ignores
    lockbook/gate constraints and non-DA markets so RHPF underperformance can be
    separated from genuinely weak DA economics.
    """
    if linprog is None:
        return {"status": "unavailable_scipy_missing"}
    if hourly.empty:
        return {"status": "unavailable_empty_hourly"}
    price_col = ""
    for candidate in ("da_price", "true_da_price", "target_da_price", "real_da_price_eur_mwh"):
        if candidate in hourly.columns:
            price_col = candidate
            break
    if not price_col:
        return {"status": "unavailable_missing_true_da_price"}
    prices = pd.to_numeric(hourly[price_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = prices.notna()
    if not bool(valid.any()):
        return {"status": "unavailable_nonfinite_true_da_price"}
    prices = prices[valid].astype(float).reset_index(drop=True)
    n = len(prices)
    dt_h = _summary_float(summary, ("time_step_hours", "dt_h"), 1.0)
    p_max = _summary_float(summary, ("p_max_mw", "battery_power_mw", "power_mw"), 10.0)
    eta_in = _summary_float(summary, ("eta_in", "efficiency_in"), 0.9487)
    eta_out = _summary_float(summary, ("eta_out", "efficiency_out"), 0.9487)
    deg = _summary_float(summary, ("degradation_cost_eur_mwh", "deg_eur_mwh"), 15.0)
    trans = _summary_float(summary, ("transaction_cost_eur_mwh", "trans_eur_mwh"), 1.0)
    soc_min = _summary_float(summary, ("soc_min_mwh", "battery_soc_min_mwh"), 2.0)
    soc_max = _summary_float(summary, ("soc_max_mwh", "battery_soc_max_mwh"), 18.0)
    soc_start = _summary_float(summary, ("initial_soc_mwh", "soc_init_mwh"), 10.0)
    soc_target = _summary_float(summary, ("final_soc_target_mwh", "soc_target_end_mwh"), 10.0)
    c = np.zeros(2 * n, dtype=float)
    for i, price in enumerate(prices.to_numpy(dtype=float)):
        buy_profit = -price * dt_h - trans * dt_h - deg * eta_in * dt_h
        sell_profit = price * dt_h - trans * dt_h - deg * dt_h / max(eta_out, 1e-12)
        c[2 * i] = -buy_profit
        c[2 * i + 1] = -sell_profit
    bounds = [(0.0, p_max) for _ in range(2 * n)]
    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    charge_coef = eta_in * dt_h
    sell_coef = -dt_h / max(eta_out, 1e-12)
    for k in range(n):
        row = np.zeros(2 * n, dtype=float)
        for j in range(k + 1):
            row[2 * j] = charge_coef
            row[2 * j + 1] = sell_coef
        a_ub.append(row.copy())
        b_ub.append(soc_max - soc_start)
        a_ub.append(-row.copy())
        b_ub.append(soc_start - soc_min)
    final_row = np.zeros(2 * n, dtype=float)
    for j in range(n):
        final_row[2 * j] = charge_coef
        final_row[2 * j + 1] = sell_coef
    a_ub.append(-final_row.copy())
    b_ub.append(soc_start - soc_target)
    res = linprog(
        c,
        A_ub=np.vstack(a_ub),
        b_ub=np.asarray(b_ub, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not bool(res.success):
        return {"status": "linprog_failed", "message": str(getattr(res, "message", ""))}
    x = np.asarray(res.x, dtype=float)
    buy_mwh = float(np.sum(x[0::2]) * dt_h)
    sell_mwh = float(np.sum(x[1::2]) * dt_h)
    final_soc = float(soc_start + np.dot(final_row, x))
    return {
        "status": "ok",
        "pnl_eur": float(-res.fun),
        "buy_mwh": buy_mwh,
        "sell_mwh": sell_mwh,
        "final_soc_mwh": final_soc,
        "price_source": price_col,
        "assumptions": "DA-only full-horizon LP; ignores gates, lockbooks, aux, and non-DA markets",
    }


def _scenario_audit(scenario_dir: Path, max_gates: int) -> tuple[bool, list[str]]:
    lines: list[str] = []
    summary_path = scenario_dir / "backtest_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    lines.append(f"\nScenario: {scenario_dir}")
    lines.append(
        "summary: "
        f"simulation_valid={summary.get('simulation_valid', 'NA')} "
        f"thesis_reportable={summary.get('thesis_reportable', 'NA')} "
        f"invalid_reason={summary.get('invalid_reason', '') or '-'}"
    )

    hourly = _read_table(scenario_dir / "backtest_hourly.parquet")
    naive = _read_table(scenario_dir / "naive_hourly.parquet")
    rhpf = _read_table(scenario_dir / "rolling_pf_hourly.parquet")
    lines.append("DA path totals:")
    for label, df, prefix in [
        ("model", hourly, "real"),
        ("naive", naive, "naive"),
        ("rhpf", rhpf, "perfect_foresight"),
    ]:
        totals = _path_da_totals(df, prefix) if not df.empty else {}
        lines.append(
            f"  {label:>5}: "
            f"submitted={totals.get('submitted_buy', 0.0):.3f}/{totals.get('submitted_sell', 0.0):.3f} "
            f"accepted={totals.get('accepted_buy', 0.0):.3f}/{totals.get('accepted_sell', 0.0):.3f} "
            f"realized={totals.get('realized_buy', 0.0):.3f}/{totals.get('realized_sell', 0.0):.3f}"
        )
    component_rows: list[dict[str, object]] = []
    for label, df, prefix in [
        ("model", hourly, "real"),
        ("naive", naive, "naive"),
        ("rhpf", rhpf, "perfect_foresight"),
    ]:
        component_source = hourly if any(col.startswith(f"{prefix}_") for col in hourly.columns) else df
        comps = _pnl_components(component_source, prefix)
        if comps:
            component_rows.append({"path": label, **comps})
    if component_rows:
        component_df = pd.DataFrame(component_rows)
        lines.append("DA PnL component audit:")
        lines.append(component_df.to_string(index=False))
        if {"model", "naive"}.issubset(set(component_df["path"].astype(str))):
            model_row = component_df.loc[component_df["path"].astype(str) == "model"].iloc[0]
            naive_row = component_df.loc[component_df["path"].astype(str) == "naive"].iloc[0]
            gap = float(model_row["pnl_eur"]) - float(naive_row["pnl_eur"])
            spread_gap = float(model_row["da_pnl_eur"]) - float(naive_row["da_pnl_eur"])
            id_gap = float(model_row["id_pnl_eur"]) - float(naive_row["id_pnl_eur"])
            lines.append(
                "Model minus naive: "
                f"total={gap:.2f} EUR, DA_PnL={spread_gap:.2f} EUR, ID_PnL={id_gap:.2f} EUR"
            )
    forecast = _forecast_quality(hourly)
    if str(forecast.get("status", "")) == "ok":
        lines.append(
            "Model DA forecast quality: "
            f"rows={float(forecast['rows']):.0f} "
            f"MAE={float(forecast['mae_eur_mwh']):.2f} EUR/MWh "
            f"bias={float(forecast['bias_eur_mwh']):.2f} EUR/MWh "
            f"corr={float(forecast['corr']):.3f} "
            f"target={forecast.get('true_col')}"
        )
        if "naive_source_min_lag_hours" in forecast:
            lines.append(
                "Naive DA source lineage: "
                f"min_lag={float(forecast['naive_source_min_lag_hours']):.1f}h "
                f"median_lag={float(forecast.get('naive_source_median_lag_hours', np.nan)):.1f}h "
                f"fallback_rows={float(forecast.get('naive_source_fallback_rows', np.nan)):.0f} "
                f"modes={forecast.get('naive_source_modes', 'unknown')}"
            )
    else:
        lines.append(f"Model DA forecast quality: unavailable status={forecast.get('status')}")
    oracle = _diagnostic_global_da_oracle(hourly, summary)
    if str(oracle.get("status", "")) == "ok":
        lines.append(
            "Diagnostic global DA oracle: "
            f"pnl={float(oracle['pnl_eur']):.2f} "
            f"buy/sell={float(oracle['buy_mwh']):.3f}/{float(oracle['sell_mwh']):.3f} "
            f"final_soc={float(oracle['final_soc_mwh']):.3f} "
            f"price_source={oracle.get('price_source')} "
            f"note={oracle.get('assumptions')}"
        )
    else:
        lines.append(f"Diagnostic global DA oracle: unavailable status={oracle.get('status')}")

    debug_path = scenario_dir / "da_precommit_debug.csv"
    plan_path = scenario_dir / "backtest_plan_history.parquet"
    debug = _read_table(debug_path)
    source_name = "da_precommit_debug.csv"
    if debug.empty:
        debug = _read_table(plan_path)
        source_name = "backtest_plan_history.parquet"
    if debug.empty:
        lines.append("DA gate audit: unavailable (no da_precommit_debug.csv or backtest_plan_history.parquet)")
        return True, lines

    if "source_snapshot_utc" not in debug.columns:
        for alt in ["snapshot_time_utc", "timestamp_utc"]:
            if alt in debug.columns:
                debug["source_snapshot_utc"] = debug[alt]
                break
    if "source_snapshot_utc" not in debug.columns:
        debug["source_snapshot_utc"] = "unknown"

    debug["_raw_buy"] = _num(debug, "da_candidate_buy_mw")
    debug["_raw_sell"] = _num(debug, "da_candidate_sell_mw")
    debug["_sized_buy"] = _num(debug, "da_sized_candidate_buy_mw")
    debug["_sized_sell"] = _num(debug, "da_sized_candidate_sell_mw")
    debug["_postlock_buy"] = _num(debug, "da_postlock_candidate_buy_mw")
    debug["_postlock_sell"] = _num(debug, "da_postlock_candidate_sell_mw")
    debug["_selected_buy"] = _num(debug, "da_selected_lockable_buy_mw")
    debug["_selected_sell"] = _num(debug, "da_selected_lockable_sell_mw")
    debug["_accepted_buy"] = _num(debug, "da_accepted_buy_mw")
    debug["_accepted_sell"] = _num(debug, "da_accepted_sell_mw")

    gate_rows: list[dict[str, object]] = []
    for gate, group in debug.groupby("source_snapshot_utc", dropna=False):
        row = {
            "source_snapshot_utc": str(gate),
            "raw_buy_mwh": float(group["_raw_buy"].sum()),
            "raw_sell_mwh": float(group["_raw_sell"].sum()),
            "sized_buy_mwh": float(group["_sized_buy"].sum()),
            "sized_sell_mwh": float(group["_sized_sell"].sum()),
            "postlock_buy_mwh": float(group["_postlock_buy"].sum()),
            "postlock_sell_mwh": float(group["_postlock_sell"].sum()),
            "selected_buy_mwh": float(group["_selected_buy"].sum()),
            "selected_sell_mwh": float(group["_selected_sell"].sum()),
            "accepted_buy_mwh": float(group["_accepted_buy"].sum()),
            "accepted_sell_mwh": float(group["_accepted_sell"].sum()),
            "candidate_pnl_eur": _first_num(group, ["candidate_selection_pnl_eur", "sized_candidate_pnl_eur"]),
            "selected_pnl_eur": _first_num(group, ["da_selected_lockable_pnl_eur", "selected_lockable_pnl_eur"]),
            "incumbent_pnl_eur": _first_num(group, ["incumbent_selection_pnl_eur", "no_trade_pnl"]),
            "terminal_sensitive": _first_num(group, ["da_terminal_sensitive_window"]),
            "terminal_reason": _first_text(group, ["da_terminal_sensitive_reason"], "unknown"),
            "terminal_shortfall_mwh": _first_num(group, ["da_terminal_shortfall_internal_mwh", "da_postlock_terminal_shortfall_mwh"]),
            "recovery_available": _first_num(group, ["da_terminal_recovery_cost_estimate_available", "da_postlock_terminal_recovery_cost_estimate_available"]),
            "zero_reason": _first_text(group, ["da_zero_reason", "zero_reason"], ""),
            "volume_loss_stage": _first_text(group, ["da_volume_loss_stage"], ""),
            "volume_loss_reason": _first_text(group, ["da_volume_loss_reason", "da_final_selection_reason"], ""),
            "bid_sizer_status": _first_text(group, ["da_bid_sizer_status"], ""),
        }
        row["candidate_minus_incumbent_pnl_eur"] = (
            float(row["candidate_pnl_eur"] - row["incumbent_pnl_eur"])
            if np.isfinite(float(row["candidate_pnl_eur"])) and np.isfinite(float(row["incumbent_pnl_eur"]))
            else float("nan")
        )
        stage, reason = _classify_gate(pd.Series(row))
        row["classified_stage"] = stage
        row["classified_reason"] = reason
        gate_rows.append(row)

    gates = pd.DataFrame(gate_rows)
    active = gates[(gates["raw_buy_mwh"] + gates["raw_sell_mwh"]) > TOL].copy()
    counts = active["classified_stage"].value_counts(dropna=False).to_dict() if not active.empty else {}
    lines.append(f"DA gate audit source: {source_name}")
    lines.append(f"active_da_candidate_gates={len(active)} loss_stage_counts={counts}")
    if not active.empty:
        display_cols = [
            "source_snapshot_utc",
            "raw_buy_mwh",
            "raw_sell_mwh",
            "sized_buy_mwh",
            "sized_sell_mwh",
            "selected_buy_mwh",
            "selected_sell_mwh",
            "accepted_buy_mwh",
            "accepted_sell_mwh",
            "candidate_minus_incumbent_pnl_eur",
            "terminal_sensitive",
            "terminal_reason",
            "classified_stage",
            "classified_reason",
        ]
        lines.append("Top DA candidate gates:")
        lines.append(active[display_cols].head(max_gates).to_string(index=False))
    inconsistent = active[
        (active["classified_stage"] != "selected")
        & (active["candidate_minus_incumbent_pnl_eur"] > TOL)
        & (active["terminal_sensitive"].fillna(0.0) < 0.5)
    ]
    if not inconsistent.empty:
        lines.append("WARNING: profitable ordinary-window candidates lost before lockbook:")
        lines.append(
            inconsistent[
                [
                    "source_snapshot_utc",
                    "candidate_minus_incumbent_pnl_eur",
                    "classified_stage",
                    "classified_reason",
                ]
            ]
            .head(max_gates)
            .to_string(index=False)
        )
    return True, lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run root or scenario directory")
    parser.add_argument("--max-gates", type=int, default=20)
    args = parser.parse_args()

    scenarios = _discover_scenarios(args.run_dir)
    if not scenarios:
        print(f"FAIL: no scenario folders found under {args.run_dir}")
        return 2
    all_ok = True
    all_lines: list[str] = []
    for scenario in scenarios:
        ok, lines = _scenario_audit(scenario, max(1, int(args.max_gates)))
        all_ok = all_ok and ok
        all_lines.extend(lines)
    print(f"DA decision pipeline audit: {'PASS' if all_ok else 'FAIL'} scenarios_checked={len(scenarios)}")
    print("\n".join(all_lines))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
