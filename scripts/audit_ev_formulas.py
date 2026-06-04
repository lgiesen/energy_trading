#!/usr/bin/env python3
"""Audit exported EV and settlement formula consistency for one backtest scenario.

The script reads a scenario directory containing ``backtest_hourly.parquet`` or
``backtest_hourly.csv`` and independently rebuilds formula identities from the
exported diagnostic columns. It does not run optimization or settlement.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TOL = 1e-6


def _read_hourly(scenario_dir: Path) -> tuple[pd.DataFrame, Path]:
    candidates = [
        scenario_dir / "backtest_hourly.parquet",
        scenario_dir / "backtest_hourly.csv",
    ]
    for path in candidates:
        if path.exists():
            if path.suffix == ".parquet":
                return pd.read_parquet(path), path
            return pd.read_csv(path), path
    nested = list(scenario_dir.rglob("backtest_hourly.parquet")) + list(
        scenario_dir.rglob("backtest_hourly.csv")
    )
    if len(nested) == 1:
        path = nested[0]
        if path.suffix == ".parquet":
            return pd.read_parquet(path), path
        return pd.read_csv(path), path
    if not nested:
        raise FileNotFoundError(f"No backtest_hourly.parquet/csv found under {scenario_dir}")
    raise FileNotFoundError(
        "Multiple backtest_hourly files found; pass a concrete scenario directory. "
        f"matches={[str(p) for p in nested[:20]]}"
    )


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _max_abs(values: Iterable[pd.Series | np.ndarray | float]) -> float:
    arrs: list[np.ndarray] = []
    for value in values:
        if isinstance(value, pd.Series):
            arrs.append(pd.to_numeric(value, errors="coerce").to_numpy(dtype=float))
        elif isinstance(value, np.ndarray):
            arrs.append(value.astype(float, copy=False).ravel())
        else:
            arrs.append(np.array([float(value)], dtype=float))
    if not arrs:
        return 0.0
    arr = np.concatenate(arrs)
    arr = arr[np.isfinite(arr)]
    return float(np.max(np.abs(arr))) if arr.size else 0.0


def _bin_ids(df: pd.DataFrame) -> list[int]:
    ids: set[int] = set()
    pat = re.compile(r"^ev_rpos_coef_bin_(\d+)_eur_per_mw$")
    for col in df.columns:
        m = pat.match(str(col))
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def audit_ev_formulas(df: pd.DataFrame) -> dict[str, object]:
    bins = _bin_ids(df)
    bcm_pos_rebuild = pd.Series(0.0, index=df.index, dtype=float)
    bcm_neg_rebuild = pd.Series(0.0, index=df.index, dtype=float)
    bem_pos_rebuild = pd.Series(0.0, index=df.index, dtype=float)
    bem_neg_rebuild = pd.Series(0.0, index=df.index, dtype=float)
    bcm_coef_errors: list[pd.Series] = []
    bem_coef_errors: list[pd.Series] = []
    bcm_capacity_errors: list[pd.Series] = []

    for b in bins:
        rpos_coef = _num(df, f"ev_rpos_coef_bin_{b}_eur_per_mw")
        rneg_coef = _num(df, f"ev_rneg_coef_bin_{b}_eur_per_mw")
        bem_pos_coef = _num(df, f"ev_bem_pos_coef_bin_{b}_eur_per_mw")
        bem_neg_coef = _num(df, f"ev_bem_neg_coef_bin_{b}_eur_per_mw")

        bcm_pos_formula = (
            _num(df, f"ev_bcm_capacity_value_pos_bin_{b}")
            + _num(df, f"ev_bcm_activation_value_pos_bin_{b}")
            - _num(df, f"ev_bcm_costs_pos_bin_{b}")
        )
        bcm_neg_formula = (
            _num(df, f"ev_bcm_capacity_value_neg_bin_{b}")
            + _num(df, f"ev_bcm_activation_value_neg_bin_{b}")
            - _num(df, f"ev_bcm_costs_neg_bin_{b}")
        )
        bem_pos_formula = (
            _num(df, f"ev_bem_activation_value_pos_bin_{b}")
            - _num(df, f"ev_bem_costs_pos_bin_{b}")
        )
        bem_neg_formula = (
            _num(df, f"ev_bem_activation_value_neg_bin_{b}")
            - _num(df, f"ev_bem_costs_neg_bin_{b}")
        )
        bcm_coef_errors.extend([rpos_coef - bcm_pos_formula, rneg_coef - bcm_neg_formula])
        bem_coef_errors.extend([bem_pos_coef - bem_pos_formula, bem_neg_coef - bem_neg_formula])

        bcm_capacity_errors.extend(
            [
                _num(df, f"ev_bcm_expected_capacity_revenue_pos_bin_{b}")
                - _num(df, f"ev_bcm_capacity_value_pos_bin_{b}"),
                _num(df, f"ev_bcm_expected_capacity_revenue_neg_bin_{b}")
                - _num(df, f"ev_bcm_capacity_value_neg_bin_{b}"),
            ]
        )
        bcm_pos_rebuild += rpos_coef * _num(df, f"reserve_pos_bin_{b}_mw")
        bcm_neg_rebuild += rneg_coef * _num(df, f"reserve_neg_bin_{b}_mw")
        bem_pos_rebuild += bem_pos_coef * _num(df, f"bem_pos_bin_{b}_mw")
        bem_neg_rebuild += bem_neg_coef * _num(df, f"bem_neg_bin_{b}_mw")

    da_charge_rebuild = _num(df, "ev_da_charge_coef_eur_per_mw") * _num(df, "charge_mw")
    da_discharge_rebuild = _num(df, "ev_da_discharge_coef_eur_per_mw") * _num(df, "discharge_mw")
    objective_rebuild = (
        _num(df, "ev_da_charge_eur")
        + _num(df, "ev_da_discharge_eur")
        + _num(df, "ev_afrr_pos_eur")
        + _num(df, "ev_afrr_neg_eur")
        + _num(df, "ev_bem_only_pos_eur")
        + _num(df, "ev_bem_only_neg_eur")
        - _num(df, "ev_slack_penalty_pos_eur")
        - _num(df, "ev_slack_penalty_neg_eur")
        + _num(df, "ev_terminal_soc_credit_eur")
    )
    activation_split_rebuild = (
        _num(df, "real_bcm_linked_activation_revenue_eur")
        + _num(df, "real_bem_only_activation_revenue_eur")
    )
    id_net_rebuild = _num(df, "real_revenue_id_eur") - _num(df, "real_cost_id_eur")
    id_net_actual = _num(df, "real_id_net_pnl_eur", default=np.nan)
    if id_net_actual.isna().all():
        id_net_actual = _num(df, "real_pnl_id_eur", default=np.nan)

    cap_neg_without_submission = pd.Series(0.0, index=df.index, dtype=float)
    if {
        "real_submitted_bcm_capacity_neg_mw",
        "real_executed_bcm_capacity_neg_mw",
    }.issubset(df.columns):
        submitted = _num(df, "real_submitted_bcm_capacity_neg_mw")
        executed = _num(df, "real_executed_bcm_capacity_neg_mw")
        cap_neg_without_submission = executed.where(submitted.abs() <= TOL, 0.0)

    metrics = {
        "rows": int(len(df)),
        "bin_count": int(len(bins)),
        "max_bcm_capacity_ev_error": _max_abs(bcm_capacity_errors),
        "max_bcm_activation_ev_error": _max_abs(bcm_coef_errors),
        "max_bem_ev_error": _max_abs(
            [
                _num(df, "ev_bem_only_pos_eur") - bem_pos_rebuild,
                _num(df, "ev_bem_only_neg_eur") - bem_neg_rebuild,
                *bem_coef_errors,
            ]
        ),
        "max_da_ev_error": _max_abs(
            [
                _num(df, "ev_da_charge_eur") - da_charge_rebuild,
                _num(df, "ev_da_discharge_eur") - da_discharge_rebuild,
            ]
        ),
        "max_id_ev_error": _max_abs([id_net_actual - id_net_rebuild])
        if not id_net_actual.isna().all()
        else float("nan"),
        "max_objective_rebuild_error": _max_abs(
            [_num(df, "ev_objective_rebuild_eur") - objective_rebuild]
        ),
        "max_settlement_revenue_split_error": _max_abs(
            [_num(df, "real_revenue_activation_eur") - activation_split_rebuild]
        ),
        "max_bcm_pos_selected_ev_error": _max_abs([_num(df, "ev_afrr_pos_eur") - bcm_pos_rebuild]),
        "max_bcm_neg_selected_ev_error": _max_abs([_num(df, "ev_afrr_neg_eur") - bcm_neg_rebuild]),
        "max_executed_bcm_neg_capacity_without_submission_mw": _max_abs([cap_neg_without_submission]),
    }
    metrics["status"] = "ok" if all(
        (not np.isfinite(v)) or abs(float(v)) <= TOL
        for k, v in metrics.items()
        if k.startswith("max_")
    ) else "failed"
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_dir", type=Path, help="Directory containing backtest_hourly.parquet/csv.")
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    scenario_dir = args.scenario_dir.resolve()
    hourly, hourly_path = _read_hourly(scenario_dir)
    metrics = audit_ev_formulas(hourly)
    metrics["scenario_dir"] = str(scenario_dir)
    metrics["hourly_path"] = str(hourly_path)

    out_csv = args.out_csv or (scenario_dir / "ev_formula_audit.csv")
    out_json = args.out_json or (scenario_dir / "ev_formula_audit.json")
    pd.DataFrame([metrics]).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[OK] EV formula audit: {out_csv} | {out_json}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
