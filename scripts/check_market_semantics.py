from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TOL_DEFAULT = 1e-6

DA_QTY_COLS = [
    "real_submitted_da_buy_mw",
    "real_submitted_da_sell_mw",
    "real_executed_charge_mw",
    "real_executed_discharge_mw",
    "real_da_buy_mwh",
    "real_da_sell_mwh",
    "da_charge_mw",
    "da_discharge_mw",
]
DA_PNL_HOURLY_COLS = ["real_revenue_da_eur", "real_cost_da_eur"]
DA_PNL_PERF_COLS = ["da_gross_revenue_eur", "da_gross_cost_eur", "da_net_revenue_eur", "da_pnl_eur"]

BCM_CAP_QTY_COLS = [
    "submitted_bcm_capacity_pos_mw",
    "submitted_bcm_capacity_neg_mw",
    "real_submitted_bcm_capacity_pos_mw",
    "real_submitted_bcm_capacity_neg_mw",
    "executed_bcm_capacity_pos_mw",
    "executed_bcm_capacity_neg_mw",
    "real_submitted_afrr_pos_mw",
    "real_submitted_afrr_neg_mw",
    "submitted_afrr_pos_mw",
    "submitted_afrr_neg_mw",
    "real_executed_reserve_pos_mw",
    "real_executed_reserve_neg_mw",
    "executed_reserve_pos_mw",
    "executed_reserve_neg_mw",
]
BCM_PNL_HOURLY_COLS = ["real_revenue_capacity_eur", "real_bcm_linked_activation_revenue_eur"]
BCM_PNL_PERF_COLS = [
    "bcm_capacity_revenue_eur",
    "bcm_linked_activation_revenue_eur",
    "bcm_strategy_total_revenue_eur",
    "bcm_pnl_eur",
]

BEM_QTY_COLS = [
    "real_bem_only_submitted_pos_mw",
    "real_bem_only_submitted_neg_mw",
    "real_bem_only_executed_pos_mwh",
    "real_bem_only_executed_neg_mwh",
    "bem_only_submitted_pos_mw",
    "bem_only_submitted_neg_mw",
    "bem_only_executed_pos_mwh",
    "bem_only_executed_neg_mwh",
]
BEM_PNL_HOURLY_COLS = ["real_bem_only_activation_revenue_eur"]
BEM_PNL_PERF_COLS = ["bem_activation_revenue_eur", "bem_net_revenue_eur", "bem_pnl_eur"]

AFRR_PNL_HOURLY_COLS = ["real_revenue_activation_eur", "real_revenue_capacity_eur"]
AFRR_PNL_PERF_COLS = ["afrr_activation_revenue_eur", "afrr_total_net_revenue_eur", "afrr_pnl_eur"]

ID_QTY_COLS = [
    "real_id_buy_mwh",
    "real_id_sell_mwh",
    "real_id_charge_mw",
    "real_id_discharge_mw",
    "pending_id_buy_mwh",
    "pending_id_sell_mwh",
]
ID_PNL_HOURLY_COLS = ["real_revenue_id_eur", "real_cost_id_eur", "real_id_pnl_eur"]
ID_PNL_PERF_COLS = ["id_gross_revenue_eur", "id_gross_cost_eur", "id_net_revenue_eur", "id_recourse_pnl_eur"]
ID_REASON_COLS = ["id_recourse_reason", "real_id_recourse_reason", "pending_id_recourse_reason", "id_reason_code"]
ID_BUY_PRICE_COLS = ["real_id_buy_price_eur_per_mwh", "id_buy_price_eur_per_mwh"]
ID_SELL_PRICE_COLS = ["real_id_sell_price_eur_per_mwh", "id_sell_price_eur_per_mwh"]

HEADROOM_VIOLATION_COLS = ["real_headroom_violation_pos_mwh", "real_headroom_violation_neg_mwh"]
HEADROOM_MARGIN_COLS = ["real_headroom_margin_pos_mwh", "real_headroom_margin_neg_mwh"]

BCM_BLOCK_ID_COLS = ["bcm_capacity_block_id"]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_table(base: Path, stem: str) -> pd.DataFrame:
    parquet = base / f"{stem}.parquet"
    csv = base / f"{stem}.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()


def _find_hourly_path(base: Path) -> Path | None:
    for name in ("backtest_hourly.parquet", "backtest_hourly.csv"):
        p = base / name
        if p.exists():
            return p
    return None


def _read_hourly(base: Path) -> pd.DataFrame:
    p = _find_hourly_path(base)
    if p is None:
        return pd.DataFrame()
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def _parse_command_line_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _to_num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _abs_total(df: pd.DataFrame, cols: list[str]) -> float:
    total = 0.0
    for col in cols:
        if col in df.columns:
            total += float(_to_num_series(df, col).fillna(0.0).abs().sum())
    return total


def _abs_activity_magnitude(hourly: pd.DataFrame, perf_row: dict[str, Any], hourly_cols: list[str], perf_cols: list[str]) -> float:
    mags = [0.0]
    mags.append(_abs_total(hourly, hourly_cols))
    for col in perf_cols:
        if col in perf_row:
            v = pd.to_numeric(pd.Series([perf_row.get(col)]), errors="coerce").iloc[0]
            if pd.notna(v):
                mags.append(abs(float(v)))
    return float(max(mags))


def _perf_value(perf_row: dict[str, Any], key: str) -> float | None:
    if key not in perf_row:
        return None
    v = pd.to_numeric(pd.Series([perf_row.get(key)]), errors="coerce").iloc[0]
    if pd.isna(v):
        return None
    return float(v)


def _severity(include_invalid: bool, warn_only: bool, hard: bool = True) -> str:
    if not hard:
        return "warning"
    if include_invalid or warn_only:
        return "warning"
    return "hard"


def _add_violation(
    violations: list[dict[str, Any]],
    *,
    severity: str,
    check_group: str,
    check_name: str,
    strategy: str,
    scenario: str,
    column: str,
    value: Any,
    tolerance: float,
    message: str,
    path: Path,
) -> None:
    violations.append(
        {
            "severity": severity,
            "check_group": check_group,
            "check_name": check_name,
            "strategy": strategy,
            "scenario": scenario,
            "column": column,
            "value": value,
            "tolerance": tolerance,
            "message": message,
            "path": str(path),
        }
    )


def _infer_model(path: Path, summary: dict[str, Any], perf_row: dict[str, Any]) -> str:
    for key in ("model_name", "model_key"):
        val = summary.get(key) or perf_row.get(key)
        if val:
            return str(val)
    text = str(path).lower()
    for label in ("xgb", "xgboost", "tft", "linear", "rlqr"):
        if label in text:
            return "xgb" if label == "xgboost" else ("linear" if label == "rlqr" else label)
    return "unknown"


def _infer_quantile(path: Path, summary: dict[str, Any], perf_row: dict[str, Any]) -> str:
    for key in ("quantile_pair", "scenario"):
        val = summary.get(key) or perf_row.get(key)
        if val:
            return str(val)
    return path.name


def _infer_strategy(path: Path, summary: dict[str, Any], perf_row: dict[str, Any]) -> str:
    val = str(summary.get("trading_strategy") or perf_row.get("trading_strategy") or "").strip().lower()
    if val:
        return val
    parts = [p.lower() for p in path.parts]
    for strategy in ("multi", "da_only", "afrr_only", "bcm_only", "bem_only"):
        if strategy in parts:
            return strategy
    return "unknown"


def _scenario_record_from_summary(summary_path: Path) -> dict[str, Any] | None:
    scenario_output_dir = summary_path.parent.resolve()
    hourly_path = _find_hourly_path(scenario_output_dir)
    if hourly_path is None:
        return None
    summary = _read_json(summary_path)
    perf_df = _read_table(scenario_output_dir, "performance_metrics")
    perf_row = perf_df.iloc[0].to_dict() if not perf_df.empty else {}
    return {
        "scenario_output_dir": scenario_output_dir,
        "summary_path": summary_path.resolve(),
        "hourly_path": hourly_path.resolve(),
        "summary": summary,
        "performance_row": perf_row,
        "strategy": _infer_strategy(scenario_output_dir, summary, perf_row),
        "model": _infer_model(scenario_output_dir, summary, perf_row),
        "quantile_policy": _infer_quantile(scenario_output_dir, summary, perf_row),
    }


def discover_scenarios(root: Path, strategy: str = "", include_invalid: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for summary_path in root.rglob("backtest_summary.json"):
        rec = _scenario_record_from_summary(summary_path)
        if rec is None:
            continue
        key = str(rec["scenario_output_dir"])
        if key in seen:
            continue
        seen.add(key)
        strat = str(rec["strategy"]).lower()
        if strategy and strat != str(strategy).lower():
            continue
        if not include_invalid:
            s = rec["summary"]
            sim_ok = float(pd.to_numeric(pd.Series([s.get("simulation_valid", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            thesis_ok = float(pd.to_numeric(pd.Series([s.get("thesis_reportable", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            if sim_ok < 0.5 or thesis_ok < 0.5:
                continue
        records.append(rec)
    return sorted(records, key=lambda r: str(r["scenario_output_dir"]))


def _group_totals(hourly: pd.DataFrame, perf_row: dict[str, Any]) -> dict[str, float]:
    return {
        "da_quantity_abs_total": _abs_total(hourly, DA_QTY_COLS),
        "da_pnl_abs_total": _abs_activity_magnitude(hourly, perf_row, DA_PNL_HOURLY_COLS, DA_PNL_PERF_COLS),
        "bcm_quantity_abs_total": _abs_total(hourly, BCM_CAP_QTY_COLS),
        "bcm_pnl_abs_total": _abs_activity_magnitude(hourly, perf_row, BCM_PNL_HOURLY_COLS, BCM_PNL_PERF_COLS),
        "bem_quantity_abs_total": _abs_total(hourly, BEM_QTY_COLS),
        "bem_pnl_abs_total": _abs_activity_magnitude(hourly, perf_row, BEM_PNL_HOURLY_COLS, BEM_PNL_PERF_COLS),
        "id_quantity_abs_total": _abs_total(hourly, ID_QTY_COLS),
        "id_pnl_abs_total": _abs_activity_magnitude(hourly, perf_row, ID_PNL_HOURLY_COLS, ID_PNL_PERF_COLS),
        "afrr_pnl_abs_total": _abs_activity_magnitude(hourly, perf_row, AFRR_PNL_HOURLY_COLS, AFRR_PNL_PERF_COLS),
    }


def _check_market_permissions(
    *,
    strategy: str,
    scenario: str,
    path: Path,
    totals: dict[str, float],
    tol: float,
    violations: list[dict[str, Any]],
) -> None:
    def _hard(name: str, col: str, value: float, message: str) -> None:
        if value > tol:
            _add_violation(
                violations,
                severity="hard",
                check_group="market_permission",
                check_name=name,
                strategy=strategy,
                scenario=scenario,
                column=col,
                value=value,
                tolerance=tol,
                message=message,
                path=path,
            )

    if strategy == "da_only":
        _hard("forbidden_bcm_quantity", "bcm_quantity_abs_total", totals["bcm_quantity_abs_total"], "da_only must not use BCM capacity markets")
        _hard("forbidden_bem_quantity", "bem_quantity_abs_total", totals["bem_quantity_abs_total"], "da_only must not use BEM-only quantities")
        _hard("forbidden_bcm_pnl", "bcm_pnl_abs_total", totals["bcm_pnl_abs_total"], "da_only must not earn BCM revenue")
        _hard("forbidden_bem_pnl", "bem_pnl_abs_total", totals["bem_pnl_abs_total"], "da_only must not earn BEM revenue")
        _hard("forbidden_afrr_pnl", "afrr_pnl_abs_total", totals["afrr_pnl_abs_total"], "da_only must not earn aFRR capacity/activation revenue")
    elif strategy == "bcm_only":
        _hard("forbidden_da_quantity", "da_quantity_abs_total", totals["da_quantity_abs_total"], "bcm_only must not use DA")
        _hard("forbidden_da_pnl", "da_pnl_abs_total", totals["da_pnl_abs_total"], "bcm_only must not earn DA PnL")
        _hard("forbidden_bem_quantity", "bem_quantity_abs_total", totals["bem_quantity_abs_total"], "bcm_only must not use discretionary BEM-only quantity")
        _hard("forbidden_bem_pnl", "bem_pnl_abs_total", totals["bem_pnl_abs_total"], "bcm_only must not earn discretionary BEM-only PnL")
    elif strategy == "bem_only":
        _hard("forbidden_da_quantity", "da_quantity_abs_total", totals["da_quantity_abs_total"], "bem_only must not use DA")
        _hard("forbidden_da_pnl", "da_pnl_abs_total", totals["da_pnl_abs_total"], "bem_only must not earn DA PnL")
        _hard("forbidden_bcm_quantity", "bcm_quantity_abs_total", totals["bcm_quantity_abs_total"], "bem_only must not use BCM capacity")
        _hard("forbidden_bcm_pnl", "bcm_pnl_abs_total", totals["bcm_pnl_abs_total"], "bem_only must not earn BCM capacity/linked activation revenue")
    elif strategy == "afrr_only":
        _hard("forbidden_da_quantity", "da_quantity_abs_total", totals["da_quantity_abs_total"], "afrr_only must not use DA")
        _hard("forbidden_da_pnl", "da_pnl_abs_total", totals["da_pnl_abs_total"], "afrr_only must not earn DA PnL")


def _check_id_recourse(
    *,
    strategy: str,
    scenario: str,
    path: Path,
    hourly: pd.DataFrame,
    summary: dict[str, Any],
    perf_row: dict[str, Any],
    totals: dict[str, float],
    tol: float,
    require_id_reason_codes: bool,
    include_invalid: bool,
    warn_only: bool,
    violations: list[dict[str, Any]],
) -> str:
    id_mode = str(summary.get("id_recourse_mode") or perf_row.get("id_recourse_mode") or "common").strip().lower()
    if not id_mode:
        id_mode = "common"
        _add_violation(
            violations,
            severity="warning",
            check_group="id_recourse",
            check_name="missing_id_recourse_mode",
            strategy=strategy,
            scenario=scenario,
            column="id_recourse_mode",
            value="",
            tolerance=tol,
            message="id_recourse_mode missing; assumed common",
            path=path,
        )
    if id_mode == "disabled":
        if totals["id_quantity_abs_total"] > tol or totals["id_pnl_abs_total"] > tol:
            _add_violation(
                violations,
                severity=_severity(include_invalid, warn_only),
                check_group="id_recourse",
                check_name="id_disabled_nonzero",
                strategy=strategy,
                scenario=scenario,
                column="id_quantity_abs_total",
                value=max(totals["id_quantity_abs_total"], totals["id_pnl_abs_total"]),
                tolerance=tol,
                message="ID recourse disabled but nonzero ID activity detected",
                path=path,
            )
    if id_mode == "afrr_obligation_only" and strategy == "da_only" and totals["id_quantity_abs_total"] > tol:
        _add_violation(
            violations,
            severity=_severity(include_invalid, warn_only),
            check_group="id_recourse",
            check_name="id_forbidden_for_da_only",
            strategy=strategy,
            scenario=scenario,
            column="id_quantity_abs_total",
            value=totals["id_quantity_abs_total"],
            tolerance=tol,
            message="da_only must not use ID when id_recourse_mode=afrr_obligation_only",
            path=path,
        )

    if totals["id_quantity_abs_total"] > tol or totals["id_pnl_abs_total"] > tol:
        reason_cols = [c for c in ID_REASON_COLS if c in hourly.columns]
        if reason_cols:
            qty_mask = pd.Series(False, index=hourly.index)
            for col in ID_QTY_COLS:
                if col in hourly.columns:
                    qty_mask |= _to_num_series(hourly, col).fillna(0.0).abs().gt(tol)
            for col in reason_cols:
                reason = hourly[col].astype(str).str.strip().str.lower()
                bad = qty_mask & (reason.isin(["", "none", "nan", "null"]))
                if bool(bad.any()):
                    _add_violation(
                        violations,
                        severity="hard" if require_id_reason_codes else "warning",
                        check_group="id_recourse",
                        check_name="missing_id_reason_code",
                        strategy=strategy,
                        scenario=scenario,
                        column=col,
                        value=int(bad.sum()),
                        tolerance=tol,
                        message="Nonzero ID trades must have a reason code",
                        path=path,
                    )
        else:
            _add_violation(
                violations,
                severity="hard" if require_id_reason_codes else "warning",
                check_group="id_recourse",
                check_name="id_reason_code_column_missing",
                strategy=strategy,
                scenario=scenario,
                column="id_reason_code",
                value="missing",
                tolerance=tol,
                message="ID activity detected but no ID reason-code column exists",
                path=path,
            )
        for buy_col in ID_BUY_PRICE_COLS:
            for sell_col in ID_SELL_PRICE_COLS:
                if buy_col in hourly.columns and sell_col in hourly.columns:
                    buy = _to_num_series(hourly, buy_col)
                    sell = _to_num_series(hourly, sell_col)
                    bad = (buy < sell - tol).fillna(False)
                    if bool(bad.any()):
                        _add_violation(
                            violations,
                            severity="hard",
                            check_group="id_recourse",
                            check_name="id_price_relation_invalid",
                            strategy=strategy,
                            scenario=scenario,
                            column=f"{buy_col}|{sell_col}",
                            value=float((sell - buy).max()),
                            tolerance=tol,
                            message="ID buy price must be >= ID sell price when both columns exist",
                            path=path,
                        )
                    break
        if totals["id_quantity_abs_total"] <= tol and totals["id_pnl_abs_total"] > tol:
            _add_violation(
                violations,
                severity="warning",
                check_group="id_recourse",
                check_name="id_pnl_without_id_quantity",
                strategy=strategy,
                scenario=scenario,
                column="id_pnl_abs_total",
                value=totals["id_pnl_abs_total"],
                tolerance=tol,
                message="Nonzero ID PnL detected with zero ID quantity totals",
                path=path,
            )
    return id_mode


def _check_bcm_blocks(
    *,
    strategy: str,
    scenario: str,
    path: Path,
    hourly: pd.DataFrame,
    summary: dict[str, Any],
    totals: dict[str, float],
    tol: float,
    violations: list[dict[str, Any]],
) -> None:
    bcm_enabled = strategy in {"bcm_only", "afrr_only", "multi"}
    if not bcm_enabled:
        if totals["bcm_quantity_abs_total"] > tol or totals["bcm_pnl_abs_total"] > tol:
            _add_violation(
                violations,
                severity="hard",
                check_group="bcm_blocks",
                check_name="forbidden_bcm_activity",
                strategy=strategy,
                scenario=scenario,
                column="bcm_quantity_abs_total",
                value=max(totals["bcm_quantity_abs_total"], totals["bcm_pnl_abs_total"]),
                tolerance=tol,
                message="Strategy without BCM permission has nonzero BCM activity",
                path=path,
            )
        return
    bcm_active = totals["bcm_quantity_abs_total"] > tol
    block_col = next((c for c in BCM_BLOCK_ID_COLS if c in hourly.columns), "")
    if bcm_active and not block_col:
        _add_violation(
            violations,
            severity="hard",
            check_group="bcm_blocks",
            check_name="missing_bcm_block_id",
            strategy=strategy,
            scenario=scenario,
            column="bcm_capacity_block_id",
            value="missing",
            tolerance=tol,
            message="BCM activity detected but bcm_capacity_block_id is missing",
            path=path,
        )
        return
    if not block_col:
        return
    checked_cols = [c for c in BCM_CAP_QTY_COLS if c in hourly.columns]
    for col in checked_cols:
        series = _to_num_series(hourly, col)
        if series.fillna(0.0).abs().sum() <= tol:
            continue
        for _, grp in hourly.assign(_v=series).groupby(block_col, dropna=True):
            vals = pd.to_numeric(grp["_v"], errors="coerce").dropna()
            if len(vals) <= 1:
                continue
            if float(vals.max() - vals.min()) > tol:
                _add_violation(
                    violations,
                    severity="hard",
                    check_group="bcm_blocks",
                    check_name="bcm_intrablock_variation",
                    strategy=strategy,
                    scenario=scenario,
                    column=col,
                    value=float(vals.max() - vals.min()),
                    tolerance=tol,
                    message="BCM capacity values must be constant within each 4h block",
                    path=path,
                )
                break


def _check_bem_hourly(
    *,
    strategy: str,
    scenario: str,
    path: Path,
    hourly: pd.DataFrame,
    summary: dict[str, Any],
    tol: float,
    violations: list[dict[str, Any]],
) -> None:
    if hourly.empty:
        return
    pos_col = next((c for c in ["real_bem_only_submitted_pos_mw", "bem_only_submitted_pos_mw"] if c in hourly.columns), "")
    neg_col = next((c for c in ["real_bem_only_submitted_neg_mw", "bem_only_submitted_neg_mw"] if c in hourly.columns), "")
    if not pos_col or not neg_col:
        return
    pos = _to_num_series(hourly, pos_col).fillna(0.0)
    neg = _to_num_series(hourly, neg_col).fillna(0.0)
    simultaneous = pos.gt(tol) & neg.gt(tol)
    if not bool(simultaneous.any()):
        return
    cmd_args = _parse_command_line_args(summary.get("command_line_args"))
    disallowed = bool(summary.get("disallow_simultaneous_bem_only_pos_neg", cmd_args.get("disallow_simultaneous_bem_only_pos_neg", False)))
    _add_violation(
        violations,
        severity="hard" if disallowed else "warning",
        check_group="bem_hourly",
        check_name="simultaneous_bem_pos_neg",
        strategy=strategy,
        scenario=scenario,
        column=f"{pos_col}|{neg_col}",
        value=int(simultaneous.sum()),
        tolerance=tol,
        message="Simultaneous positive and negative BEM bids detected in the same hour",
        path=path,
    )


def _check_revenue_split(
    *,
    strategy: str,
    scenario: str,
    path: Path,
    hourly: pd.DataFrame,
    tol: float,
    violations: list[dict[str, Any]],
) -> None:
    required = {
        "real_revenue_activation_eur",
        "real_bcm_linked_activation_revenue_eur",
        "real_bem_only_activation_revenue_eur",
    }
    if not required.issubset(hourly.columns):
        return
    total = _to_num_series(hourly, "real_revenue_activation_eur").fillna(0.0)
    bcm = _to_num_series(hourly, "real_bcm_linked_activation_revenue_eur").fillna(0.0)
    bem = _to_num_series(hourly, "real_bem_only_activation_revenue_eur").fillna(0.0)
    err = (total - (bcm + bem)).abs()
    max_err = float(err.max()) if len(err) else 0.0
    if max_err > tol:
        _add_violation(
            violations,
            severity="hard",
            check_group="revenue_split",
            check_name="activation_revenue_split_mismatch",
            strategy=strategy,
            scenario=scenario,
            column="real_revenue_activation_eur",
            value=max_err,
            tolerance=tol,
            message="Activation revenue must equal BCM-linked plus BEM-only activation revenue",
            path=path,
        )


def _check_constraints(
    *,
    strategy: str,
    scenario: str,
    path: Path,
    hourly: pd.DataFrame,
    summary: dict[str, Any],
    perf_row: dict[str, Any],
    tol: float,
    include_invalid: bool,
    warn_only: bool,
    violations: list[dict[str, Any]],
) -> None:
    soc = next((c for c in ["real_soc_mwh", "soc_mwh"] if c in hourly.columns), "")
    soc_min = _perf_value(summary, "soc_min_mwh")
    soc_max = _perf_value(summary, "soc_max_mwh")
    if soc and soc_min is not None:
        min_soc = float(_to_num_series(hourly, soc).min())
        if min_soc < soc_min - tol:
            _add_violation(violations, severity="hard", check_group="constraints", check_name="soc_below_min", strategy=strategy, scenario=scenario, column=soc, value=min_soc, tolerance=tol, message="Physical SoC fell below soc_min_mwh", path=path)
    if soc and soc_max is not None:
        max_soc = float(_to_num_series(hourly, soc).max())
        if max_soc > soc_max + tol:
            _add_violation(violations, severity="hard", check_group="constraints", check_name="soc_above_max", strategy=strategy, scenario=scenario, column=soc, value=max_soc, tolerance=tol, message="Physical SoC exceeded soc_max_mwh", path=path)

    for col in HEADROOM_VIOLATION_COLS:
        if col in hourly.columns:
            v = float(_to_num_series(hourly, col).fillna(0.0).max())
            if v > tol:
                _add_violation(violations, severity="hard", check_group="headroom", check_name="headroom_violation", strategy=strategy, scenario=scenario, column=col, value=v, tolerance=tol, message="Explicit headroom violation column exceeded tolerance", path=path)
    for col in HEADROOM_MARGIN_COLS:
        if col in hourly.columns:
            v = float(_to_num_series(hourly, col).fillna(0.0).min())
            if v < -tol:
                _add_violation(violations, severity="hard", check_group="headroom", check_name="negative_headroom_margin", strategy=strategy, scenario=scenario, column=col, value=v, tolerance=tol, message="Headroom margin below zero", path=path)

    p_max = _perf_value(summary, "p_max_mw")
    if p_max is not None:
        cols_pos = ["real_executed_discharge_mw", "real_bem_only_submitted_pos_mw", "real_executed_reserve_pos_mw", "real_id_discharge_mw", "real_da_discharge_mw", "da_discharge_mw"]
        cols_neg = ["real_executed_charge_mw", "real_bem_only_submitted_neg_mw", "real_executed_reserve_neg_mw", "real_id_charge_mw", "real_da_charge_mw", "da_charge_mw"]
        used_pos = [c for c in cols_pos if c in hourly.columns]
        used_neg = [c for c in cols_neg if c in hourly.columns]
        if used_pos:
            pos_stack = sum(_to_num_series(hourly, c).fillna(0.0) for c in used_pos)
            v = float((pos_stack - p_max).max())
            if v > tol:
                _add_violation(violations, severity="hard", check_group="constraints", check_name="positive_power_stack_exceeds_pmax", strategy=strategy, scenario=scenario, column="|".join(used_pos), value=v, tolerance=tol, message="Positive power stack exceeds p_max_mw", path=path)
        else:
            _add_violation(violations, severity="skipped", check_group="constraints", check_name="positive_power_stack_skipped", strategy=strategy, scenario=scenario, column="power_stack_pos", value="", tolerance=tol, message="Not enough positive-stack columns to validate p_max", path=path)
        if used_neg:
            neg_stack = sum(_to_num_series(hourly, c).fillna(0.0) for c in used_neg)
            v = float((neg_stack - p_max).max())
            if v > tol:
                _add_violation(violations, severity="hard", check_group="constraints", check_name="negative_power_stack_exceeds_pmax", strategy=strategy, scenario=scenario, column="|".join(used_neg), value=v, tolerance=tol, message="Negative power stack exceeds p_max_mw", path=path)
        else:
            _add_violation(violations, severity="skipped", check_group="constraints", check_name="negative_power_stack_skipped", strategy=strategy, scenario=scenario, column="power_stack_neg", value="", tolerance=tol, message="Not enough negative-stack columns to validate p_max", path=path)

    if "optimization_error_code" in hourly.columns:
        bad = hourly["optimization_error_code"].astype(str).str.strip().str.lower().ne("ok")
        if bool(bad.any()):
            _add_violation(
                violations,
                severity=_severity(include_invalid, warn_only),
                check_group="constraints",
                check_name="optimization_error_code_non_ok",
                strategy=strategy,
                scenario=scenario,
                column="optimization_error_code",
                value=int(bad.sum()),
                tolerance=tol,
                message="Non-ok optimization_error_code found",
                path=path,
            )
    if "is_fallback_hour" in hourly.columns:
        bad = _to_num_series(hourly, "is_fallback_hour").fillna(0.0).gt(tol)
        if bool(bad.any()):
            _add_violation(
                violations,
                severity=_severity(include_invalid, warn_only),
                check_group="constraints",
                check_name="fallback_hour_present",
                strategy=strategy,
                scenario=scenario,
                column="is_fallback_hour",
                value=int(bad.sum()),
                tolerance=tol,
                message="Fallback hour detected",
                path=path,
            )

    strict_mode = bool(summary.get("strict_simulation_validity", True))
    final_mode = str(summary.get("final_soc_mode") or perf_row.get("final_soc_mode") or "hard").strip().lower()
    final_actual = _perf_value(summary, "final_soc_actual_mwh")
    final_target = _perf_value(summary, "final_soc_target_mwh")
    constraint_ok = summary.get("final_soc_constraint_satisfied")
    terminal_repair_cost = _perf_value(summary, "terminal_soc_repair_cost_eur") or 0.0
    if strict_mode and final_mode == "hard" and final_actual is not None and final_target is not None:
        if final_actual < final_target - tol:
            _add_violation(violations, severity=_severity(include_invalid, warn_only), check_group="constraints", check_name="final_soc_below_target", strategy=strategy, scenario=scenario, column="final_soc_actual_mwh", value=final_actual, tolerance=tol, message="Final physical SoC below hard target", path=path)
        if constraint_ok is not None and not bool(constraint_ok):
            _add_violation(violations, severity=_severity(include_invalid, warn_only), check_group="constraints", check_name="final_soc_constraint_flag_false", strategy=strategy, scenario=scenario, column="final_soc_constraint_satisfied", value=constraint_ok, tolerance=tol, message="final_soc_constraint_satisfied is false in strict hard mode", path=path)
        if terminal_repair_cost > tol:
            _add_violation(violations, severity=_severity(include_invalid, warn_only), check_group="constraints", check_name="terminal_repair_cost_in_strict_hard_mode", strategy=strategy, scenario=scenario, column="terminal_soc_repair_cost_eur", value=terminal_repair_cost, tolerance=tol, message="Terminal repair cost should not be used in strict hard final SoC mode", path=path)


def _check_accounting(
    *,
    strategy: str,
    scenario: str,
    path: Path,
    hourly: pd.DataFrame,
    perf_row: dict[str, Any],
    daily_df: pd.DataFrame,
    perf_debug_df: pd.DataFrame,
    tol: float,
    violations: list[dict[str, Any]],
) -> None:
    realized_hourly = float(_to_num_series(hourly, "real_pnl_eur").fillna(0.0).sum()) if "real_pnl_eur" in hourly.columns else None
    realized_perf = _perf_value(perf_row, "realized_net_revenue_eur")
    if realized_hourly is not None and realized_perf is not None and abs(realized_hourly - realized_perf) > tol:
        _add_violation(violations, severity="hard", check_group="accounting", check_name="hourly_to_performance_net_mismatch", strategy=strategy, scenario=scenario, column="realized_net_revenue_eur", value=realized_hourly - realized_perf, tolerance=tol, message="sum(hourly.real_pnl_eur) does not match performance realized_net_revenue_eur", path=path)

    if realized_hourly is not None and not daily_df.empty and "net_revenue_eur" in daily_df.columns:
        daily_sum = float(_to_num_series(daily_df, "net_revenue_eur").fillna(0.0).sum())
        if abs(daily_sum - realized_hourly) > tol:
            _add_violation(violations, severity="hard", check_group="accounting", check_name="daily_to_hourly_net_mismatch", strategy=strategy, scenario=scenario, column="net_revenue_eur", value=daily_sum - realized_hourly, tolerance=tol, message="sum(daily.net_revenue_eur) does not match sum(hourly.real_pnl_eur)", path=path)

    component_cost_keys = [
        "realized_degradation_cost_eur",
        "realized_aux_cost_eur",
        "transaction_cost_eur",
        "offer_cost_eur",
        "penalty_cost_eur",
        "terminal_soc_repair_cost_eur",
    ]
    total_costs = _perf_value(perf_row, "total_costs_eur")
    cost_parts = [_perf_value(perf_row, k) for k in component_cost_keys]
    if total_costs is not None and all(v is not None for v in cost_parts):
        expected = float(sum(v for v in cost_parts if v is not None))
        if abs(total_costs - expected) > tol:
            _add_violation(violations, severity="hard", check_group="accounting", check_name="cost_reconciliation_failure", strategy=strategy, scenario=scenario, column="total_costs_eur", value=total_costs - expected, tolerance=tol, message="total_costs_eur does not match component costs", path=path)
    else:
        _add_violation(violations, severity="skipped", check_group="accounting", check_name="cost_reconciliation_skipped", strategy=strategy, scenario=scenario, column="total_costs_eur", value="", tolerance=tol, message="Missing cost fields for cost reconciliation", path=path)

    decomp_keys = [
        "da_gross_revenue_eur",
        "da_gross_cost_eur",
        "id_gross_revenue_eur",
        "id_gross_cost_eur",
        "afrr_capacity_revenue_eur",
        "afrr_activation_revenue_eur",
        "total_costs_eur",
        "realized_net_revenue_eur",
    ]
    if all(k in perf_row and _perf_value(perf_row, k) is not None for k in decomp_keys):
        reconstructed = (
            float(_perf_value(perf_row, "da_gross_revenue_eur") or 0.0)
            - float(_perf_value(perf_row, "da_gross_cost_eur") or 0.0)
            + float(_perf_value(perf_row, "id_gross_revenue_eur") or 0.0)
            - float(_perf_value(perf_row, "id_gross_cost_eur") or 0.0)
            + float(_perf_value(perf_row, "afrr_capacity_revenue_eur") or 0.0)
            + float(_perf_value(perf_row, "afrr_activation_revenue_eur") or 0.0)
            - float(_perf_value(perf_row, "total_costs_eur") or 0.0)
        )
        realized = float(_perf_value(perf_row, "realized_net_revenue_eur") or 0.0)
        if abs(reconstructed - realized) > tol:
            _add_violation(violations, severity="hard", check_group="accounting", check_name="component_decomposition_failure", strategy=strategy, scenario=scenario, column="realized_net_revenue_eur", value=reconstructed - realized, tolerance=tol, message="Reconstructed net PnL from components does not match realized net PnL", path=path)
    else:
        _add_violation(violations, severity="skipped", check_group="accounting", check_name="component_decomposition_skipped", strategy=strategy, scenario=scenario, column="realized_net_revenue_eur", value="", tolerance=tol, message="Missing component fields for full PnL decomposition", path=path)

    if not perf_debug_df.empty:
        checked = perf_debug_df.copy()
        if "checked_daily_to_scenario" in checked.columns:
            checked = checked.loc[pd.to_numeric(checked["checked_daily_to_scenario"], errors="coerce").fillna(0.0) > 0.5].copy()
        for col in ("daily_abs_error", "hourly_abs_error"):
            if col in checked.columns:
                bad = pd.to_numeric(checked[col], errors="coerce").fillna(0.0).abs().gt(tol)
                if bool(bad.any()):
                    _add_violation(violations, severity="hard", check_group="accounting", check_name=f"performance_debug_{col}_failure", strategy=strategy, scenario=scenario, column=col, value=float(pd.to_numeric(checked.loc[bad, col], errors="coerce").abs().max()), tolerance=tol, message="performance_metric_reconciliation_debug contains checked rows above tolerance", path=path)


def validate_scenario(
    scenario_output_dir: Path,
    *,
    tolerance: float = TOL_DEFAULT,
    require_id_reason_codes: bool = False,
    include_invalid: bool = False,
    warn_only: bool = False,
    check_constraints: bool = True,
    check_headroom: bool = True,
    check_bcm_blocks: bool = True,
    check_accounting: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    summary = _read_json(scenario_output_dir / "backtest_summary.json")
    hourly = _read_hourly(scenario_output_dir)
    perf_df = _read_table(scenario_output_dir, "performance_metrics")
    perf_row = perf_df.iloc[0].to_dict() if not perf_df.empty else {}
    daily_df = _read_table(scenario_output_dir, "daily_performance_metrics")
    perf_debug_df = _read_table(scenario_output_dir, "performance_metric_reconciliation_debug")

    strategy = _infer_strategy(scenario_output_dir, summary, perf_row)
    scenario = _infer_quantile(scenario_output_dir, summary, perf_row)
    model = _infer_model(scenario_output_dir, summary, perf_row)
    totals = _group_totals(hourly, perf_row)
    violations: list[dict[str, Any]] = []

    _check_market_permissions(strategy=strategy, scenario=scenario, path=scenario_output_dir, totals=totals, tol=tolerance, violations=violations)
    id_mode = _check_id_recourse(
        strategy=strategy,
        scenario=scenario,
        path=scenario_output_dir,
        hourly=hourly,
        summary=summary,
        perf_row=perf_row,
        totals=totals,
        tol=tolerance,
        require_id_reason_codes=require_id_reason_codes,
        include_invalid=include_invalid,
        warn_only=warn_only,
        violations=violations,
    )
    if check_bcm_blocks:
        _check_bcm_blocks(strategy=strategy, scenario=scenario, path=scenario_output_dir, hourly=hourly, summary=summary, totals=totals, tol=tolerance, violations=violations)
    if check_constraints or check_headroom:
        _check_constraints(
            strategy=strategy,
            scenario=scenario,
            path=scenario_output_dir,
            hourly=hourly,
            summary=summary,
            perf_row=perf_row,
            tol=tolerance,
            include_invalid=include_invalid,
            warn_only=warn_only,
            violations=violations,
        )
    _check_bem_hourly(strategy=strategy, scenario=scenario, path=scenario_output_dir, hourly=hourly, summary=summary, tol=tolerance, violations=violations)
    _check_revenue_split(strategy=strategy, scenario=scenario, path=scenario_output_dir, hourly=hourly, tol=tolerance, violations=violations)
    if check_accounting:
        _check_accounting(
            strategy=strategy,
            scenario=scenario,
            path=scenario_output_dir,
            hourly=hourly,
            perf_row=perf_row,
            daily_df=daily_df,
            perf_debug_df=perf_debug_df,
            tol=tolerance,
            violations=violations,
        )

    vdf = pd.DataFrame(
        violations,
        columns=["severity", "check_group", "check_name", "strategy", "scenario", "column", "value", "tolerance", "message", "path"],
    )
    hard_violations = int((vdf["severity"] == "hard").sum()) if not vdf.empty else 0
    warning_count = int((vdf["severity"] == "warning").sum()) if not vdf.empty else 0
    skipped_checks = int((vdf["severity"] == "skipped").sum()) if not vdf.empty else 0
    summary_row = {
        "scenario_output_dir": str(scenario_output_dir),
        "model": model,
        "strategy": strategy,
        "quantile_policy": scenario,
        "simulation_valid": summary.get("simulation_valid"),
        "thesis_reportable": summary.get("thesis_reportable"),
        "invalid_reason": summary.get("invalid_reason"),
        "base_strategy_id_mode": summary.get("base_strategy_id_mode"),
        "id_recourse_mode": id_mode,
        "resolved_id_mode": summary.get("resolved_id_mode", summary.get("id_mode")),
        "id_allowed": summary.get("id_allowed"),
        **totals,
        "market_permission_ok": hard_violations == 0 or not bool(vdf.loc[vdf["check_group"].eq("market_permission") & vdf["severity"].eq("hard")].any(axis=None) if not vdf.empty else False),
        "id_recourse_ok": not bool(vdf.loc[vdf["check_group"].eq("id_recourse") & vdf["severity"].eq("hard")].any(axis=None) if not vdf.empty else False),
        "bcm_block_ok": not bool(vdf.loc[vdf["check_group"].eq("bcm_blocks") & vdf["severity"].eq("hard")].any(axis=None) if not vdf.empty else False),
        "revenue_split_ok": not bool(vdf.loc[vdf["check_group"].eq("revenue_split") & vdf["severity"].eq("hard")].any(axis=None) if not vdf.empty else False),
        "constraints_ok": not bool(vdf.loc[vdf["check_group"].eq("constraints") & vdf["severity"].eq("hard")].any(axis=None) if not vdf.empty else False),
        "headroom_ok": not bool(vdf.loc[vdf["check_group"].eq("headroom") & vdf["severity"].eq("hard")].any(axis=None) if not vdf.empty else False),
        "accounting_ok": not bool(vdf.loc[vdf["check_group"].eq("accounting") & vdf["severity"].eq("hard")].any(axis=None) if not vdf.empty else False),
        "thesis_semantics_ok": hard_violations == 0,
        "hard_violation_count": hard_violations,
        "warning_count": warning_count,
        "skipped_check_count": skipped_checks,
    }
    return summary_row, vdf


def run_market_semantics_validation(
    *,
    simulation_root: Path,
    out_dir: Path,
    strategy: str = "",
    include_invalid: bool = False,
    tolerance: float = TOL_DEFAULT,
    warn_only: bool = False,
    require_id_reason_codes: bool = False,
    check_constraints: bool = True,
    check_headroom: bool = True,
    check_bcm_blocks: bool = True,
    check_accounting: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = discover_scenarios(simulation_root, strategy=strategy, include_invalid=include_invalid)
    summary_rows: list[dict[str, Any]] = []
    violations_all: list[pd.DataFrame] = []
    for rec in scenarios:
        row, vdf = validate_scenario(
            Path(rec["scenario_output_dir"]),
            tolerance=tolerance,
            require_id_reason_codes=require_id_reason_codes,
            include_invalid=include_invalid,
            warn_only=warn_only,
            check_constraints=check_constraints,
            check_headroom=check_headroom,
            check_bcm_blocks=check_bcm_blocks,
            check_accounting=check_accounting,
        )
        summary_rows.append(row)
        if not vdf.empty:
            violations_all.append(vdf)
    summary_df = pd.DataFrame(summary_rows)
    violations_df = pd.concat(violations_all, ignore_index=True) if violations_all else pd.DataFrame(
        columns=["severity", "check_group", "check_name", "strategy", "scenario", "column", "value", "tolerance", "message", "path"]
    )
    summary_csv = out_dir / "market_semantics_summary.csv"
    violations_csv = out_dir / "market_semantics_violations.csv"
    report_json = out_dir / "market_semantics_report.json"
    summary_df.to_csv(summary_csv, index=False)
    violations_df.to_csv(violations_csv, index=False)
    report = {
        "simulation_root": str(simulation_root),
        "scenarios_checked": int(len(summary_df)),
        "hard_violations": int((violations_df["severity"] == "hard").sum()) if not violations_df.empty else 0,
        "warnings": int((violations_df["severity"] == "warning").sum()) if not violations_df.empty else 0,
        "skipped_checks": int((violations_df["severity"] == "skipped").sum()) if not violations_df.empty else 0,
        "overall_ok": bool(violations_df.empty or not (violations_df["severity"] == "hard").any()),
        "output_paths": {
            "market_semantics_summary_csv": str(summary_csv),
            "market_semantics_violations_csv": str(violations_csv),
            "market_semantics_report_json": str(report_json),
        },
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Validate thesis-grade market semantics on existing battery simulation outputs.")
    ap.add_argument("--simulation-root", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--strategy", default="")
    ap.add_argument("--include-invalid", action="store_true")
    ap.add_argument("--tolerance", type=float, default=TOL_DEFAULT)
    ap.add_argument("--warn-only", action="store_true")
    ap.add_argument("--require-id-reason-codes", action="store_true")
    ap.add_argument("--check-constraints", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--check-headroom", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--check-bcm-blocks", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--check-accounting", action=argparse.BooleanOptionalAction, default=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    simulation_root = Path(args.simulation_root)
    if not simulation_root.exists():
        raise FileNotFoundError(f"Simulation root not found: {simulation_root}")
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else simulation_root / "market_semantics_validation"
    report = run_market_semantics_validation(
        simulation_root=simulation_root,
        out_dir=out_dir,
        strategy=args.strategy,
        include_invalid=bool(args.include_invalid),
        tolerance=float(args.tolerance),
        warn_only=bool(args.warn_only),
        require_id_reason_codes=bool(args.require_id_reason_codes),
        check_constraints=bool(args.check_constraints),
        check_headroom=bool(args.check_headroom),
        check_bcm_blocks=bool(args.check_bcm_blocks),
        check_accounting=bool(args.check_accounting),
    )
    return 0 if report["overall_ok"] or bool(args.warn_only) else 1


if __name__ == "__main__":
    raise SystemExit(main())
