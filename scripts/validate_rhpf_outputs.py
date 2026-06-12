from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_TOL_EUR = 1e-4
DEFAULT_TOL_MWH = 1e-6


@dataclass
class ScenarioResult:
    scenario_dir: Path
    status: str = "PASS"
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    selected_row: dict[str, Any] = field(default_factory=dict)
    solver_row: dict[str, Any] = field(default_factory=dict)
    revenue_row: dict[str, Any] = field(default_factory=dict)
    suspicious: dict[str, Any] = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.status = "FAIL"
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _safe_float(v: Any, default: float = math.nan) -> float:
    try:
        out = float(pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0])
    except Exception:
        return default
    return out if pd.notna(out) else default


def _safe_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    return str(v)


def _near(a: float, b: float, tol: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _discover_scenarios(root: Path) -> list[Path]:
    if (root / "backtest_summary.json").exists():
        return [root]
    return sorted({p.parent for p in root.rglob("backtest_summary.json")})


def _sum_col(df: pd.DataFrame, *cols: str) -> float:
    for col in cols:
        if col in df.columns:
            return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
    return 0.0


def _max_col(df: pd.DataFrame, *cols: str) -> float:
    vals: list[pd.Series] = []
    for col in cols:
        if col in df.columns:
            vals.append(pd.to_numeric(df[col], errors="coerce"))
    if not vals:
        return math.nan
    merged = pd.concat(vals, axis=0).dropna()
    return float(merged.max()) if not merged.empty else math.nan


def _infer_strategy(scenario_dir: Path, summary: dict[str, Any]) -> str:
    for key in ("trading_strategy", "strategy", "strat"):
        val = _safe_str(summary.get(key)).strip().lower()
        if val:
            return val
    parts = [p.lower() for p in scenario_dir.parts]
    for candidate in ("multi", "bcm", "bem", "da", "afrr"):
        if candidate in parts:
            return candidate
    return "unknown"


def _print_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    if not rows:
        print("  <empty>")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    widths = {key: max(len(str(key)), *(len(_format_cell(row.get(key, ""))) for row in rows)) for key in keys}
    print("  " + " | ".join(str(key).ljust(widths[key]) for key in keys))
    print("  " + "-+-".join("-" * widths[key] for key in keys))
    for row in rows:
        print("  " + " | ".join(_format_cell(row.get(key, "")).ljust(widths[key]) for key in keys))


def _format_cell(v: Any) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.6g}"
    return str(v)


def _validate_aliases(result: ScenarioResult, summary: dict[str, Any], tol: float) -> None:
    reported = _safe_float(summary.get("rolling_pf_reported_total_pnl_eur"))
    reported_available = _safe_float(summary.get("rolling_pf_reported_available"), 0.0) >= 0.5
    aliases = [
        "rolling_perfect_foresight_same_rules_total_pnl_eur",
        "same_rules_rolling_pf_total_pnl_eur",
        "rhpf_total_pnl_eur",
    ]
    for alias in aliases:
        val = _safe_float(summary.get(alias))
        if reported_available:
            if not math.isfinite(val):
                result.fail(f"summary alias {alias} is missing/non-finite while reported RHPF is available")
            elif not _near(val, reported, tol):
                result.fail(f"summary alias {alias}={val:.6f} differs from reported={reported:.6f}")
        elif math.isfinite(val):
            result.fail(f"summary alias {alias} is finite although reported raw-solver RHPF is unavailable")


def _validate_solver_accounting(result: ScenarioResult, summary: dict[str, Any], tol: float) -> None:
    available = _safe_float(summary.get("rolling_pf_available"), 0.0) >= 0.5
    reported_available = _safe_float(summary.get("rolling_pf_reported_available"), 0.0) >= 0.5
    solver_total = _safe_float(summary.get("rolling_pf_solver_total_pnl_eur"))
    solver_component = _safe_float(summary.get("rolling_pf_solver_component_pnl_eur"))
    solver_error = _safe_float(summary.get("rolling_pf_solver_pnl_balance_error_eur"))
    solver_ok = _safe_float(summary.get("rolling_pf_solver_pnl_balance_ok"), 0.0) >= 0.5
    reason = (
        _safe_str(summary.get("rolling_pf_solver_invalid_reason"))
        or _safe_str(summary.get("rolling_pf_solver_infeasible_reason"))
        or _safe_str(summary.get("rolling_pf_failure_reason"))
    )
    if available:
        if not math.isfinite(solver_total):
            result.warn("raw solver total PnL is missing/non-finite although RHPF is available")
        if math.isfinite(solver_total) and math.isfinite(solver_component):
            derived_error = solver_total - solver_component
            if math.isfinite(solver_error) and abs(derived_error - solver_error) > max(tol, 1e-6):
                result.warn(
                    "solver pnl balance error field does not match solver_total - solver_component "
                    f"({solver_error:.6f} vs {derived_error:.6f})"
                )
            if solver_ok and abs(derived_error) > tol:
                result.fail(
                    "raw solver PnL balance is marked OK but solver_total - solver_component "
                    f"is {derived_error:.6f}"
                )
        if not solver_ok and (not reason or reason in {"none", "not_evaluated"}) and math.isfinite(solver_error) and abs(solver_error) > tol:
            result.fail("raw solver PnL balance is not OK and no explicit solver failure/infeasible reason exists")
        if reported_available and not solver_ok:
            result.fail("reported RHPF is available although raw solver PnL balance is not OK")


def _validate_selected_accounting(result: ScenarioResult, summary: dict[str, Any], tol: float) -> None:
    available = _safe_float(summary.get("rolling_pf_available"), 0.0) >= 0.5
    selected_total = _safe_float(summary.get("rolling_pf_selected_total_pnl_eur"))
    selected_component = _safe_float(summary.get("rolling_pf_selected_component_pnl_eur"))
    selected_error = _safe_float(summary.get("rolling_pf_selected_pnl_balance_error_eur"))
    selected_ok = _safe_float(summary.get("rolling_pf_selected_pnl_balance_ok"), 0.0) >= 0.5
    if available:
        if not math.isfinite(selected_total):
            result.fail("selected RHPF total PnL is missing/non-finite")
        if not selected_ok:
            result.fail("selected RHPF PnL balance is not OK")
        if math.isfinite(selected_total) and math.isfinite(selected_component) and math.isfinite(selected_error):
            derived_error = selected_total - selected_component
            if abs(derived_error - selected_error) > max(tol, 1e-6):
                result.warn(
                    "selected pnl balance error field does not match selected_total - selected_component "
                    f"({selected_error:.6f} vs {derived_error:.6f})"
                )


def _validate_incumbent_flags(
    result: ScenarioResult,
    summary: dict[str, Any],
    *,
    allow_fallback_benchmark: bool,
) -> None:
    available = _safe_float(summary.get("rolling_pf_available"), 0.0) >= 0.5
    reported_available = _safe_float(summary.get("rolling_pf_reported_available"), 0.0) >= 0.5
    reported_is_solver = _safe_float(summary.get("rolling_pf_reported_is_solver"), 0.0) >= 0.5
    flags = {
        "solver": _safe_float(summary.get("rolling_pf_selected_is_solver"), 0.0),
        "realized_path": _safe_float(summary.get("rolling_pf_selected_is_realized_path_fallback"), 0.0),
        "no_market": _safe_float(summary.get("rolling_pf_selected_is_no_market_fallback"), 0.0),
    }
    if available:
        active = [name for name, val in flags.items() if val >= 0.5]
        if len(active) != 1:
            result.fail(f"expected exactly one selected-incumbent flag, got {active}")
        elif active[0] != "solver":
            msg = (
                f"not pure solver RHPF: selected fallback={active[0]}, "
                f"reason={_safe_str(summary.get('rolling_pf_incumbent_selection_reason'))}"
            )
            if allow_fallback_benchmark:
                result.warn(msg)
            else:
                result.fail(msg)
    if reported_available and not reported_is_solver:
        result.fail("reported RHPF is available but rolling_pf_reported_is_solver is not 1")


def _validate_da(result: ScenarioResult, summary: dict[str, Any]) -> None:
    model_da = abs(_safe_float(summary.get("real_da_buy_mwh"), 0.0)) + abs(_safe_float(summary.get("real_da_sell_mwh"), 0.0))
    reported_solver = _safe_float(summary.get("rolling_pf_reported_is_solver"), 0.0) >= 0.5
    pf_da_qty = abs(_safe_float(summary.get("rolling_pf_da_realized_buy_mwh"), 0.0)) + abs(
        _safe_float(summary.get("rolling_pf_da_realized_sell_mwh"), 0.0)
    )
    pf_da_money = abs(_safe_float(summary.get("rolling_pf_da_revenue_eur"), 0.0)) + abs(
        _safe_float(summary.get("rolling_pf_da_cost_eur"), 0.0)
    )
    if model_da > DEFAULT_TOL_MWH and reported_solver and pf_da_qty <= DEFAULT_TOL_MWH and pf_da_money <= DEFAULT_TOL_EUR:
        result.warn("model has DA trades but selected raw solver RHPF has zero DA quantity/revenue/cost")
    if _safe_float(summary.get("rolling_pf_da_price_taker_mode"), 0.0) >= 0.5:
        rejected = abs(_safe_float(summary.get("rolling_pf_da_price_rejected_buy_mwh"), 0.0)) + abs(
            _safe_float(summary.get("rolling_pf_da_price_rejected_sell_mwh"), 0.0)
        )
        if rejected > DEFAULT_TOL_MWH:
            result.fail("DA price_taker RHPF has nonzero DA price rejection")


def _validate_bcm(result: ScenarioResult, summary: dict[str, Any], hourly: pd.DataFrame) -> None:
    missing_summary = _safe_float(summary.get("bcm_capacity_price_missing_for_awarded_capacity_count"), 0.0)
    if missing_summary > 0.0 or "bcm_capacity_price_missing_for_awarded_capacity" in _safe_str(summary.get("invalid_reason")):
        result.fail("bcm_capacity_price_missing_for_awarded_capacity is present")
    if hourly.empty:
        result.warn("cannot validate BCM hourly capacity price/revenue because backtest_hourly.parquet is missing")
        return
    awarded_pos = _sum_col(hourly, "perfect_foresight_locked_bcm_capacity_pos_mw", "perfect_foresight_executed_bcm_capacity_pos_mw")
    awarded_neg = _sum_col(hourly, "perfect_foresight_locked_bcm_capacity_neg_mw", "perfect_foresight_executed_bcm_capacity_neg_mw")
    awarded = awarded_pos + awarded_neg
    if awarded > DEFAULT_TOL_MWH:
        price_max = _max_col(
            hourly,
            "perfect_foresight_settlement_cap_bid_price_pos_eur_mw",
            "perfect_foresight_settlement_cap_bid_price_neg_eur_mw",
            "perfect_foresight_bcm_capacity_bid_price_pos_eur_per_mw_h",
            "perfect_foresight_bcm_capacity_bid_price_neg_eur_per_mw_h",
        )
        revenue = _sum_col(hourly, "perfect_foresight_revenue_capacity_eur", "perfect_foresight_bcm_capacity_revenue_eur")
        if not math.isfinite(price_max) or price_max <= 0.0:
            result.fail("RHPF awarded/locked BCM MW exists but no positive capacity bid price is visible")
        if revenue <= DEFAULT_TOL_EUR:
            result.fail("RHPF awarded/locked BCM MW exists but capacity revenue is zero/non-positive")


def _validate_bem(result: ScenarioResult, hourly: pd.DataFrame, tol: float) -> None:
    if hourly.empty:
        result.warn("cannot validate BEM activation reconciliation because backtest_hourly.parquet is missing")
        return
    needed = [
        "perfect_foresight_bem_only_pos_activation_mwh",
        "perfect_foresight_bem_only_neg_activation_mwh",
        "perfect_foresight_executed_afrr_act_pos_bin_0_price_eur_mwh",
        "perfect_foresight_executed_afrr_act_neg_bin_0_price_eur_mwh",
        "perfect_foresight_bem_only_activation_revenue_eur",
    ]
    if not all(c in hourly.columns for c in needed):
        result.warn("BEM activation mWh/price/revenue columns are incomplete; skipped direct BEM price*mWh check")
        return
    pos = pd.to_numeric(hourly[needed[0]], errors="coerce").fillna(0.0)
    neg = pd.to_numeric(hourly[needed[1]], errors="coerce").fillna(0.0)
    pos_price = pd.to_numeric(hourly[needed[2]], errors="coerce").fillna(0.0)
    neg_price = pd.to_numeric(hourly[needed[3]], errors="coerce").fillna(0.0)
    actual = float(pd.to_numeric(hourly[needed[4]], errors="coerce").fillna(0.0).sum())
    approx = float((pos * pos_price + neg * neg_price).sum())
    if abs(actual - approx) > max(tol, 1e-4):
        result.warn(f"BEM activation revenue does not reconcile to simple mWh*price approximation: actual={actual:.6f}, approx={approx:.6f}")


def _validate_multi(result: ScenarioResult, hourly: pd.DataFrame, tol: float) -> None:
    if hourly.empty:
        result.warn("cannot validate multi-market component double counting because backtest_hourly.parquet is missing")
        return
    pf_da = _sum_col(hourly, "perfect_foresight_revenue_da_eur") - _sum_col(hourly, "perfect_foresight_cost_da_eur")
    pf_capacity = _sum_col(hourly, "perfect_foresight_revenue_capacity_eur")
    pf_activation = _sum_col(hourly, "perfect_foresight_revenue_activation_eur")
    pf_bem_only = _sum_col(hourly, "perfect_foresight_bem_only_activation_revenue_eur")
    pf_bcm_linked = _sum_col(hourly, "perfect_foresight_bcm_linked_activation_revenue_eur")
    if abs((pf_bem_only + pf_bcm_linked) - pf_activation) > max(tol, 1e-4) and abs(pf_activation) > tol:
        result.warn(
            "RHPF activation revenue split does not reconcile: "
            f"activation={pf_activation:.6f}, bem_only+bcm_linked={pf_bem_only + pf_bcm_linked:.6f}"
        )
    if abs(pf_da) > 0 and abs(pf_capacity) > 0 and abs(pf_activation) > 0:
        result.info.append("multi-market RHPF has DA, BCM capacity, and activation revenue components; inspect splits for thesis reporting")


def _collect_rows(result: ScenarioResult, summary: dict[str, Any], hourly: pd.DataFrame) -> None:
    result.selected_row = {
        "scenario": str(result.scenario_dir),
        "selected": _safe_str(summary.get("rolling_pf_selected_incumbent")),
        "selected_total": _safe_float(summary.get("rolling_pf_selected_total_pnl_eur")),
        "reported_total": _safe_float(summary.get("rolling_pf_reported_total_pnl_eur")),
        "reported_available": _safe_float(summary.get("rolling_pf_reported_available"), 0.0),
        "reported_reason": _safe_str(summary.get("rolling_pf_reported_invalid_reason")),
        "solver_flag": _safe_float(summary.get("rolling_pf_selected_is_solver"), 0.0),
        "realized_path_flag": _safe_float(summary.get("rolling_pf_selected_is_realized_path_fallback"), 0.0),
        "no_market_flag": _safe_float(summary.get("rolling_pf_selected_is_no_market_fallback"), 0.0),
        "reason": _safe_str(summary.get("rolling_pf_incumbent_selection_reason")),
    }
    result.solver_row = {
        "scenario": str(result.scenario_dir),
        "solver_total": _safe_float(summary.get("rolling_pf_solver_total_pnl_eur")),
        "solver_component": _safe_float(summary.get("rolling_pf_solver_component_pnl_eur")),
        "solver_balance_error": _safe_float(summary.get("rolling_pf_solver_pnl_balance_error_eur")),
        "solver_invalid_reason": _safe_str(summary.get("rolling_pf_solver_invalid_reason")),
        "reported_total": _safe_float(summary.get("rolling_pf_reported_total_pnl_eur")),
        "selected_total": _safe_float(summary.get("rolling_pf_selected_total_pnl_eur")),
        "no_market": _safe_float(summary.get("rolling_pf_no_market_incumbent_eur")),
        "realized_path": _safe_float(summary.get("rolling_pf_realized_path_incumbent_eur")),
        "solver_minus_realized": _safe_float(summary.get("rolling_pf_solver_minus_realized_path_eur")),
        "dominance_pass": _safe_float(summary.get("rolling_pf_dominance_check_pass"), 0.0),
    }
    result.revenue_row = {
        "scenario": str(result.scenario_dir),
        "realized_pnl": _safe_float(summary.get("realized_total_pnl_eur")),
        "rhpf_reported": _safe_float(summary.get("rolling_pf_reported_total_pnl_eur")),
        "rhpf_selected_diag": _safe_float(summary.get("rolling_pf_selected_total_pnl_eur")),
        "rhpf_solver": _safe_float(summary.get("rolling_pf_solver_total_pnl_eur")),
        "pf_da_gross": _safe_float(summary.get("rolling_pf_da_gross_eur")),
        "pf_bcm_capacity_rev": _sum_col(hourly, "perfect_foresight_revenue_capacity_eur") if not hourly.empty else math.nan,
        "pf_activation_rev": _sum_col(hourly, "perfect_foresight_revenue_activation_eur") if not hourly.empty else math.nan,
    }
    fields = [
        "simulation_valid",
        "thesis_reportable",
        "invalid_reason",
        "fallback_used",
        "optimization_error_code_counts",
        "rolling_pf_da_zero_trade_reason",
        "rolling_pf_da_first_rejection_reason",
        "rolling_pf_da_volume_loss_stage",
        "rolling_pf_da_volume_loss_reason",
        "rolling_pf_bcm_plan_settlement_mismatch_reason",
        "rolling_pf_reported_invalid_reason",
        "bcm_capacity_price_missing_for_awarded_capacity_count",
    ]
    result.suspicious = {k: summary.get(k) for k in fields if k in summary}


def validate_scenario(scenario_dir: Path, tol: float, *, allow_fallback_benchmark: bool = False) -> ScenarioResult:
    result = ScenarioResult(scenario_dir=scenario_dir)
    summary_path = scenario_dir / "backtest_summary.json"
    summary = _read_json(summary_path)
    if not summary:
        result.fail("missing or unreadable backtest_summary.json")
        return result
    hourly = _read_parquet(scenario_dir / "backtest_hourly.parquet")
    rolling_pf = _read_parquet(scenario_dir / "rolling_pf_hourly.parquet")
    naive = _read_parquet(scenario_dir / "naive_hourly.parquet")
    if hourly.empty:
        result.warn("backtest_hourly.parquet missing or unreadable")
    if rolling_pf.empty:
        result.warn("rolling_pf_hourly.parquet missing or unreadable")
    if naive.empty:
        result.warn("naive_hourly.parquet missing or unreadable")

    _validate_aliases(result, summary, tol)
    _validate_solver_accounting(result, summary, tol)
    _validate_selected_accounting(result, summary, tol)
    _validate_incumbent_flags(result, summary, allow_fallback_benchmark=allow_fallback_benchmark)

    strategy = _infer_strategy(scenario_dir, summary)
    if "da" in strategy or strategy == "multi":
        _validate_da(result, summary)
    if "bcm" in strategy or "afrr" in strategy or strategy == "multi":
        _validate_bcm(result, summary, hourly)
    if "bem" in strategy or strategy == "multi":
        _validate_bem(result, hourly, tol)
    if strategy == "multi":
        _validate_multi(result, hourly, tol)

    _collect_rows(result, summary, hourly)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate RHPF semantics/accounting for simulation outputs.")
    parser.add_argument("run_dir", type=Path, help="Simulation run directory or direct scenario directory")
    parser.add_argument("--tol-eur", type=float, default=DEFAULT_TOL_EUR, help="EUR tolerance for accounting checks")
    parser.add_argument(
        "--allow-fallback-benchmark",
        action="store_true",
        help="Treat realized_path/no_market selected incumbents as warnings instead of failures.",
    )
    args = parser.parse_args()

    scenarios = _discover_scenarios(args.run_dir)
    if not scenarios:
        print(f"FAIL: no backtest_summary.json found below {args.run_dir}")
        return 2

    results = [
        validate_scenario(path, args.tol_eur, allow_fallback_benchmark=bool(args.allow_fallback_benchmark))
        for path in scenarios
    ]
    overall = "PASS" if all(r.status == "PASS" for r in results) else "FAIL"
    print(f"RHPF validation: {overall}")
    print(f"scenarios_checked={len(results)}")

    _print_table("RHPF selected incumbent table", [r.selected_row for r in results])
    _print_table("Raw solver vs selected incumbent table", [r.solver_row for r in results])
    _print_table("Strategy-specific revenue/cost decomposition", [r.revenue_row for r in results])
    _print_table("Top suspicious fields", [{"scenario": str(r.scenario_dir), **r.suspicious} for r in results])

    for r in results:
        print(f"\n[{r.status}] {r.scenario_dir}")
        if r.failures:
            print("  failures:")
            for msg in r.failures:
                print(f"    - {msg}")
        if r.warnings:
            print("  warnings:")
            for msg in r.warnings:
                print(f"    - {msg}")
        if r.info:
            print("  info:")
            for msg in r.info:
                print(f"    - {msg}")

    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
