#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow execution as `python scripts/generate_strategy_diagnostics.py`.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy_trading.visualization.style import THESIS_PALETTE, apply_geo_style


POINT_POLICIES = ["p10-p10", "p30-p30", "p50-p50", "p70-p70", "p90-p90"]
SYMMETRIC_POLICIES = ["p10-p90", "p30-p70"]
ASYMMETRIC_POLICIES = ["p10-p30", "p30-p50", "p50-p70", "p70-p90"]
DEFAULT_POLICIES = POINT_POLICIES + SYMMETRIC_POLICIES + ASYMMETRIC_POLICIES
EXPECTED_OUTPUTS = [
    "quantile_sensitivity_point.png",
    "quantile_sensitivity_symmetric_interval.png",
    "quantile_sensitivity_asymmetric_interval.png",
    "quantile_profit_point.tex",
    "quantile_profit_symmetric_interval.tex",
    "quantile_profit_asymmetric_interval.tex",
    "detailed_performance_point.tex",
    "detailed_performance_symmetric_interval.tex",
    "detailed_performance_asymmetric_interval.tex",
    "revenue_cost_decomposition_point.png",
    "revenue_cost_decomposition_symmetric_interval.png",
    "revenue_cost_decomposition_asymmetric_interval.png",
    "costs_lossday_point.png",
    "costs_lossday_symmetric_interval.png",
    "costs_lossday_asymmetric_interval.png",
    "daily_profit_distribution.png",
    "accuracy_profit_scatter.png",
    "accuracy_profit_table.tex",
    "risk_robustness.tex",
    "id_penalty_sensitivity_point.png",
    "id_penalty_sensitivity_symmetric_interval.png",
    "id_penalty_sensitivity_asymmetric_interval.png",
    "cumulative_pnl_by_policy.png",
]


@dataclass
class SimulationArtifacts:
    metrics_rows: list[dict[str, Any]]
    daily_rows: list[dict[str, Any]]
    scenario_records: list[dict[str, Any]]
    warnings: list[str]


def _safe_float(v: Any, default: float = np.nan) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latex_escape(x: Any) -> str:
    s = str(x)
    repl = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def _infer_model_label(path_like: str) -> str:
    s = str(path_like).lower()
    if "xgb" in s or "xgboost" in s:
        return "xgboost"
    if "linear" in s or "rlqr" in s:
        return "linear"
    if "tft" in s:
        return "tft"
    return "unknown"


def _categorize_quantile_policy(q: str) -> str:
    q = str(q)
    if q in POINT_POLICIES:
        return "point"
    if q in SYMMETRIC_POLICIES:
        return "symmetric_interval"
    if q in ASYMMETRIC_POLICIES:
        return "asymmetric_interval"
    return "unknown"


def _compute_shared_ylim(dfs: list[pd.DataFrame], value_cols: list[str], pad_frac: float = 0.05) -> tuple[float, float]:
    vals: list[float] = []
    for df in dfs:
        if df is None or df.empty:
            continue
        for c in value_cols:
            if c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                vals.extend(s.tolist())
    if not vals:
        return -1.0, 1.0
    lo, hi = min(vals), max(vals)
    if np.isclose(lo, hi):
        pad = max(1.0, abs(lo) * pad_frac + 1e-6)
        return lo - pad, hi + pad
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def _parse_qpair(name: str) -> tuple[str, str]:
    s = str(name)
    m = re.search(r"(p\d{2})[-_](p\d{2})", s)
    if m:
        return m.group(1), m.group(2)
    return "", ""


def _normalize_quantile_policy(name: str) -> str:
    ql, qh = _parse_qpair(name)
    if ql and qh:
        return f"{ql}-{qh}"
    return str(name)


def _expected_layout_hint(simulation_root: Path) -> str:
    return (
        f"No simulation artifacts found under {simulation_root}. "
        "Expected one of these layouts: "
        "1) root/backtest_summary.json + root/backtest_hourly.parquet|csv; "
        "2) root/multi/p50_p50/backtest_summary.json; "
        "3) root/xgb_multi_p50-p50/multi/p50_p50/backtest_summary.json; "
        "4) any recursive folder containing backtest_summary.json paired with backtest_hourly.parquet|csv."
    )


def _scenario_record_from_pair(summary_path: Path, hourly_path: Path, source: str) -> dict[str, Any]:
    summary = _load_json(summary_path)
    scenario_output_dir = summary_path.parent.resolve()
    scenario_name = str(summary.get("scenario", scenario_output_dir.name))
    quantile_policy = _normalize_quantile_policy(
        str(summary.get("quantile_pair", "") or summary.get("scenario", "") or scenario_output_dir.name)
    )
    if quantile_policy == str(scenario_output_dir.name):
        quantile_policy = _normalize_quantile_policy(str(summary_path.parent.parent.name) + "_" + str(summary_path.parent.name))
        if quantile_policy == str(summary_path.parent.parent.name) + "_" + str(summary_path.parent.name):
            quantile_policy = _normalize_quantile_policy(scenario_name)
    quantile_low, quantile_high = _parse_qpair(quantile_policy)
    trading_strategy = str(summary.get("trading_strategy", "") or "")
    model_value = str(summary.get("model_name", "") or summary.get("model_key", "") or "")
    if not model_value:
        model_value = _infer_model_label(str(scenario_output_dir))
    if not trading_strategy:
        parts = [p.lower() for p in scenario_output_dir.parts]
        trading_strategy = "multi" if "multi" in parts else ""
    model_key = model_value
    if model_value == "xgboost":
        model_key = "xgb"
    elif model_value == "linear":
        model_key = "linear"
    elif model_value == "tft":
        model_key = "tft"
    return {
        "source": source,
        "scenario_output_dir": str(scenario_output_dir),
        "output_dir": str(scenario_output_dir),
        "summary_path": str(summary_path.resolve()),
        "hourly_path": str(hourly_path.resolve()),
        "scenario": scenario_name,
        "model": model_value,
        "model_key": model_key,
        "trading_strategy": trading_strategy or "unknown",
        "quantile_policy": quantile_policy,
        "quantile_low": quantile_low,
        "quantile_high": quantile_high,
        "quantile_category": _categorize_quantile_policy(quantile_policy),
        "simulation_valid": _safe_float(summary.get("simulation_valid", np.nan)),
        "thesis_reportable": _safe_float(summary.get("thesis_reportable", np.nan)),
        "invalid_reason": str(summary.get("invalid_reason", "") or ""),
    }


def _pick_first_numeric(df: pd.DataFrame, cols: list[str], default: float = 0.0) -> pd.Series:
    for c in cols:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _resolve_output_dir(raw: str, sweep_parent: Path) -> Path | None:
    p = Path(str(raw))
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    p2 = (sweep_parent / p)
    if p2.exists():
        return p2.resolve()
    return None


def discover_simulation_artifacts(
    simulation_root: Path,
    scenario_dirs: list[Path] | None = None,
    include_invalid: bool = False,
) -> SimulationArtifacts:
    warnings: list[str] = []
    metrics_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    scenario_records: list[dict[str, Any]] = []
    roots = [Path(p) for p in scenario_dirs] if scenario_dirs else [simulation_root]
    seen: set[tuple[str, str]] = set()
    searched_patterns = [
        "backtest_summary.json + backtest_hourly.parquet",
        "backtest_summary.json + backtest_hourly.csv",
        "root/multi/pXX_pYY/backtest_summary.json",
        "root/<job>/multi/pXX-pYY/backtest_summary.json",
    ]

    for root in roots:
        root = root.resolve()
        direct_summary = root / "backtest_summary.json"
        direct_hourly_parq = root / "backtest_hourly.parquet"
        direct_hourly_csv = root / "backtest_hourly.csv"
        if direct_summary.exists() and (direct_hourly_parq.exists() or direct_hourly_csv.exists()):
            hp = direct_hourly_parq if direct_hourly_parq.exists() else direct_hourly_csv
            rec = _scenario_record_from_pair(direct_summary, hp, "direct_root")
            key = (rec["summary_path"], rec["hourly_path"])
            if key not in seen:
                scenario_records.append(rec)
                seen.add(key)
        for sp in root.rglob("backtest_summary.json"):
            out_dir = sp.parent
            hp_parq = out_dir / "backtest_hourly.parquet"
            hp_csv = out_dir / "backtest_hourly.csv"
            if not hp_parq.exists() and not hp_csv.exists():
                continue
            hp = hp_parq if hp_parq.exists() else hp_csv
            rec = _scenario_record_from_pair(sp, hp, "recursive_discovery")
            key = (rec["summary_path"], rec["hourly_path"])
            if key in seen:
                continue
            scenario_records.append(rec)
            seen.add(key)

    if not scenario_records:
        candidates = sorted([str(p.relative_to(simulation_root)) for p in simulation_root.rglob("*") if p.is_file()])
        raise FileNotFoundError(
            _expected_layout_hint(simulation_root)
            + f" searched_patterns={searched_patterns} files_found={len(candidates)} first_candidates={candidates[:30]}"
        )

    scenario_records = sorted(
        scenario_records,
        key=lambda r: (
            str(r.get("model", "")),
            str(r.get("trading_strategy", "")),
            str(r.get("quantile_policy", "")),
            str(r.get("scenario_output_dir", "")),
        ),
    )
    if not include_invalid:
        excluded = [
            r for r in scenario_records
            if not (np.isfinite(float(r.get("simulation_valid", np.nan))) and float(r.get("simulation_valid", 0.0)) >= 0.5
                    and np.isfinite(float(r.get("thesis_reportable", np.nan))) and float(r.get("thesis_reportable", 0.0)) >= 0.5)
        ]
        if excluded:
            warnings.append(
                "excluded non-reportable scenarios: "
                + "; ".join(
                    f"{r.get('model')}|{r.get('trading_strategy')}|{r.get('quantile_policy')} reason={r.get('invalid_reason','')}"
                    for r in excluded
                )
            )

    return SimulationArtifacts(
        metrics_rows=metrics_rows,
        daily_rows=daily_rows,
        scenario_records=scenario_records,
        warnings=warnings,
    )


def _filter_time(df: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None, ts_col: str = "timestamp_utc") -> pd.DataFrame:
    if ts_col not in df.columns:
        return df
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= ts >= start
    if end is not None:
        mask &= ts <= end
    return df.loc[mask].copy()


def _infer_analysis_duration_days(ts: pd.Series) -> tuple[float, float, float]:
    ts2 = pd.to_datetime(ts, utc=True, errors="coerce").dropna().sort_values()
    if ts2.empty:
        return 0.0, 0.0, 0.0
    if len(ts2) == 1:
        dt_h = 1.0
    else:
        dh = ts2.diff().dropna().dt.total_seconds() / 3600.0
        dt_h = float(dh.median()) if len(dh) else 1.0
        if not np.isfinite(dt_h) or dt_h <= 0:
            dt_h = 1.0
    hours = float(len(ts2) * dt_h)
    days = max(1e-9, hours / 24.0)
    annualization = 365.0 / days
    return hours, days, annualization


def _compute_scenario_metrics(art: SimulationArtifacts, start: pd.Timestamp | None, end: pd.Timestamp | None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    hourly_by_output: dict[str, pd.DataFrame] = {}

    for r in art.scenario_records:
        sp = Path(r["summary_path"])
        hp = Path(r["hourly_path"])
        summary = _load_json(sp)
        if hp.suffix == ".parquet":
            hourly = pd.read_parquet(hp)
        else:
            hourly = pd.read_csv(hp)
        hourly = _filter_time(hourly, start, end)
        if hourly.empty:
            continue
        hourly_by_output[str(r["output_dir"])] = hourly.copy()

        ts = pd.to_datetime(hourly.get("timestamp_utc"), utc=True, errors="coerce")
        pnl_h = _pick_first_numeric(hourly, ["real_pnl_eur", "pnl_eur"], 0.0)
        realized = float(pnl_h.sum())
        analysis_hours, n_days, annualization_factor = _infer_analysis_duration_days(ts)
        annualized = float(realized * annualization_factor)
        qpolicy = _normalize_quantile_policy(str(r.get("quantile_policy", "") or r.get("scenario", "")))
        qlow, qhigh = _parse_qpair(qpolicy)
        if not qlow:
            qlow = str(r.get("quantile_low", ""))
            qhigh = str(r.get("quantile_high", ""))
            qpolicy = f"{qlow}-{qhigh}" if qlow and qhigh else str(r.get("scenario", ""))

        da = _pick_first_numeric(hourly, ["real_revenue_da_eur", "da_revenue_eur"], 0.0) - _pick_first_numeric(hourly, ["real_cost_da_eur", "da_cost_eur"], 0.0)
        bcm = _pick_first_numeric(hourly, ["real_revenue_capacity_eur", "bcm_capacity_revenue_eur"], 0.0)
        bem = _pick_first_numeric(hourly, ["real_revenue_activation_eur", "bem_net_revenue_eur"], 0.0)
        idr = _pick_first_numeric(hourly, ["real_id_pnl_eur", "id_recourse_pnl_eur"], 0.0)
        deg = _pick_first_numeric(hourly, ["real_degradation_cost_eur", "degradation_cost_eur"], 0.0)
        pen = _pick_first_numeric(hourly, ["real_penalty_eur", "penalty_cost_eur"], 0.0)
        aux = _pick_first_numeric(hourly, ["real_aux_cost_eur", "aux_cost_eur"], 0.0)

        pnl_daily = pd.DataFrame({"ts": ts, "pnl": pnl_h}).dropna()
        pnl_daily["day"] = pnl_daily["ts"].dt.floor("D")
        daily = pnl_daily.groupby("day", as_index=False)["pnl"].sum().rename(columns={"pnl": "daily_pnl_eur"})
        for drec in daily.to_dict(orient="records"):
            daily_rows.append(
                {
                    "model": str(r.get("model", _infer_model_label(r["output_dir"]))),
                    "quantile_policy": qpolicy,
                    "trading_strategy": str(summary.get("trading_strategy", "multi")).lower(),
                    "date": str(drec["day"]),
                    "daily_pnl_eur": float(drec["daily_pnl_eur"]),
                }
            )

        row = {
            "model": str(r.get("model", _infer_model_label(r["output_dir"]))),
            "model_key": str(r.get("model_key", _infer_model_label(r["output_dir"]))),
            "trading_strategy": str(r.get("trading_strategy", summary.get("trading_strategy", "multi"))).lower(),
            "quantile_policy": qpolicy,
            "quantile_low": qlow,
            "quantile_high": qhigh,
            "quantile_category": _categorize_quantile_policy(qpolicy),
            "simulation_valid": _safe_float(summary.get("simulation_valid", 1.0), 1.0),
            "thesis_reportable": _safe_float(summary.get("thesis_reportable", 1.0), 1.0),
            "invalid_reason": str(summary.get("invalid_reason", "")),
            "realized_net_profit_eur": realized,
            "annualized_realized_net_profit_eur": annualized,
            "da_pnl_eur": float(da.sum()),
            "afrr_pnl_eur": float((bcm + bem).sum()),
            "bcm_pnl_eur": float(bcm.sum()),
            "bem_pnl_eur": float(bem.sum()),
            "id_recourse_pnl_eur": float(idr.sum()),
            "degradation_cost_eur": float(deg.sum()),
            "penalty_cost_eur": float(pen.sum()),
            "aux_cost_eur": float(aux.sum()),
            "total_costs_eur": float((deg + pen + aux).sum()),
            "analysis_hours": analysis_hours,
            "analysis_days": n_days,
            "annualization_factor": annualization_factor,
            "cycles": _safe_float(summary.get("total_equivalent_full_cycles", np.nan)),
            "equivalent_full_cycles": _safe_float(summary.get("total_equivalent_full_cycles", np.nan)),
            "id_recourse_event_count": float((_pick_first_numeric(hourly, ["real_id_charge_mw"], 0.0).abs() + _pick_first_numeric(hourly, ["real_id_discharge_mw"], 0.0).abs() > 1e-9).sum()),
            "output_dir": str(r["output_dir"]),
        }

        if daily.empty:
            row.update({
                "loss_day_share": np.nan,
                "mean_daily_profit_eur": np.nan,
                "median_daily_profit_eur": np.nan,
                "worst_day_eur": np.nan,
                "worst_week_eur": np.nan,
                "daily_pnl_q05_eur": np.nan,
                "cvar_5_eur": np.nan,
                "max_drawdown_eur": np.nan,
                "profit_volatility_eur": np.nan,
            })
        else:
            dvals = pd.to_numeric(daily["daily_pnl_eur"], errors="coerce").dropna()
            row["loss_day_share"] = float((dvals < 0.0).mean())
            row["mean_daily_profit_eur"] = float(dvals.mean())
            row["median_daily_profit_eur"] = float(dvals.median())
            row["worst_day_eur"] = float(dvals.min())
            q05 = float(dvals.quantile(0.05))
            row["daily_pnl_q05_eur"] = q05
            row["cvar_5_eur"] = float(dvals[dvals <= q05].mean()) if (dvals <= q05).any() else q05
            row["profit_volatility_eur"] = float(dvals.std(ddof=0))
            cum = dvals.cumsum()
            peak = cum.cummax()
            dd = cum - peak
            row["max_drawdown_eur"] = float(dd.min())
            w = daily.copy()
            w["week"] = pd.to_datetime(w["day"], utc=True).dt.tz_localize(None).dt.to_period("W").astype(str)
            row["worst_week_eur"] = float(pd.to_numeric(w.groupby("week")["daily_pnl_eur"].sum(), errors="coerce").min())

        for c in [
            "da_pnl_eur", "afrr_pnl_eur", "bcm_pnl_eur", "bem_pnl_eur", "id_recourse_pnl_eur",
            "degradation_cost_eur", "penalty_cost_eur", "aux_cost_eur", "total_costs_eur",
        ]:
            row[f"annualized_{c}"] = float(_safe_float(row[c], 0.0) * annualization_factor)
        row["annualized_id_recourse_cost_eur"] = float(max(0.0, -row["annualized_id_recourse_pnl_eur"]))

        rows.append(row)

    if art.metrics_rows:
        warnings.append("aggregate performance metrics were discovered but scenario-level recomputation was preferred")

    return pd.DataFrame(rows), pd.DataFrame(daily_rows), hourly_by_output, warnings


def _normalize_metrics(metrics: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    alias = {
        "scenario": "quantile_policy",
        "annualized_realized_pnl_eur": "annualized_realized_net_profit_eur",
        "annualized_realized_net_revenue_eur": "annualized_realized_net_profit_eur",
        "realized_total_pnl_eur": "realized_net_profit_eur",
        "pnl_real_eur": "realized_net_profit_eur",
        "total_id_pnl_eur": "id_recourse_pnl_eur",
        "id_net_revenue_eur": "id_recourse_pnl_eur",
        "da_net_revenue_eur": "da_pnl_eur",
        "bcm_capacity_revenue_eur": "bcm_pnl_eur",
        "bem_net_revenue_eur": "bem_pnl_eur",
        "total_equivalent_full_cycles": "equivalent_full_cycles",
    }
    out = metrics.copy()
    for src, dst in alias.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]

    required = ["model", "quantile_policy", "annualized_realized_net_profit_eur", "realized_net_profit_eur"]
    miss = [c for c in required if c not in out.columns]
    if miss:
        raise ValueError(f"missing required metrics columns: {miss}; available={list(out.columns)}")

    for c in [
        "da_pnl_eur", "afrr_pnl_eur", "bcm_pnl_eur", "bem_pnl_eur", "id_recourse_pnl_eur",
        "degradation_cost_eur", "penalty_cost_eur", "aux_cost_eur", "loss_day_share",
        "mean_daily_profit_eur", "median_daily_profit_eur", "worst_day_eur", "worst_week_eur",
        "daily_pnl_q05_eur", "cvar_5_eur", "max_drawdown_eur", "profit_volatility_eur",
        "equivalent_full_cycles", "id_recourse_event_count", "constraint_pressure_index",
        "mean_pinball_loss", "gate_specific_pinball_loss",
        "annualized_da_pnl_eur", "annualized_afrr_pnl_eur", "annualized_bcm_pnl_eur", "annualized_bem_pnl_eur",
        "annualized_id_recourse_pnl_eur", "annualized_degradation_cost_eur", "annualized_penalty_cost_eur",
        "annualized_aux_cost_eur", "annualized_total_costs_eur", "annualized_id_recourse_cost_eur",
    ]:
        if c not in out.columns:
            out[c] = np.nan
            warnings.append(f"optional metric missing; filled NaN: {c}")
    if "quantile_category" not in out.columns:
        out["quantile_category"] = out["quantile_policy"].astype(str).map(_categorize_quantile_policy)
    if "trading_strategy" not in out.columns:
        out["trading_strategy"] = "multi"
    return out


def _comparative_diagnosis(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    for model, g in df.groupby("model"):
        g2 = g.sort_values("realized_net_profit_eur", ascending=False)
        best = g2.iloc[0]
        if "constraint_pressure_index" in g.columns:
            constraint_pressure = _safe_float(g["constraint_pressure_index"].mean(), 0.0)
        else:
            constraint_pressure = np.nan
            warnings.append(f"constraint_pressure_index missing for model={model}; recommendation fallback applied")
        rows.append(
            {
                "model": model,
                "best_quantile_policy": str(best.get("quantile_policy", "")),
                "best_realized_pnl_eur": _safe_float(best.get("realized_net_profit_eur")),
                "mean_realized_pnl_eur": _safe_float(g["realized_net_profit_eur"].mean()),
                "mean_constraint_pressure_index": constraint_pressure,
                "recommendation": (
                    "Use conservative interval policy"
                    if (np.isfinite(constraint_pressure) and constraint_pressure > 0.0)
                    else "Use best-performing policy with standard guardrails"
                ),
            }
        )
    return pd.DataFrame(rows)


def _latex_table(df: pd.DataFrame, caption: str, label: str, float_format: str = "{:.2f}") -> str:
    cols = list(df.columns)
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{@{}" + "l" * len(cols) + "@{}}",
        "\\toprule",
        " & ".join(_latex_escape(c) for c in cols) + " \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, (int, float, np.floating)) and pd.notna(v):
                vals.append(float_format.format(float(v)))
            else:
                sv = str(v)
                if sv.startswith("\\textbf{"):
                    vals.append(sv)
                else:
                    vals.append(_latex_escape(v))
        lines.append(" & ".join(vals) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def _plot_quantile_sensitivity(metrics: pd.DataFrame, figures_dir: Path, data_dir: Path, checklist: list[dict[str, Any]]) -> None:
    apply_geo_style()
    cats = ["point", "symmetric_interval", "asymmetric_interval"]
    grouped = {}
    for cat in cats:
        g = metrics.loc[metrics["quantile_category"] == cat].copy()
        grouped[cat] = g
    ylo, yhi = _compute_shared_ylim(list(grouped.values()), ["annualized_realized_net_profit_eur"])

    for cat in cats:
        g = grouped[cat]
        out_csv = data_dir / f"quantile_sensitivity_{cat}.csv"
        out_png = figures_dir / f"quantile_sensitivity_{cat}.png"
        g.to_csv(out_csv, index=False)
        if g.empty:
            checklist.append({"output": f"quantile_sensitivity_{cat}", "implemented": "yes", "status": "skipped_no_data", "path": "", "notes": "no data"})
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        for model, gm in g.groupby("model"):
            gm = gm.copy()
            gm["qorder"] = gm["quantile_policy"].map({q: i for i, q in enumerate(DEFAULT_POLICIES)}).fillna(999)
            gm = gm.sort_values("qorder")
            x = np.arange(len(gm))
            y = pd.to_numeric(gm["annualized_realized_net_profit_eur"], errors="coerce").fillna(0.0).to_numpy()
            ax.plot(x, y, label=str(model))
            ax.fill_between(x, 0.0, y, alpha=0.18)
            ax.set_xticks(x)
            ax.set_xticklabels(gm["quantile_policy"].astype(str), rotation=45, ha="right")
        ax.set_ylim(ylo, yhi)
        ax.set_title("Annualized Realized Net Profit under Multi-Market Strategy")
        ax.set_ylabel("Annualized net profit [EUR]")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        checklist.append({"output": f"quantile_sensitivity_{cat}", "implemented": "yes", "file_path": str(out_png), "notes": ""})


def _render_tables_and_plots(
    metrics: pd.DataFrame,
    daily: pd.DataFrame,
    hourly_by_output: dict[str, pd.DataFrame],
    *,
    figures_dir: Path,
    tables_dir: Path,
    data_dir: Path,
    warnings: list[str],
) -> tuple[int, int, list[dict[str, Any]]]:
    rep_fig_dir = figures_dir / "representative_weeks"
    rep_data_dir = data_dir / "representative_weeks"
    for p in [figures_dir, tables_dir, data_dir, rep_fig_dir, rep_data_dir]:
        p.mkdir(parents=True, exist_ok=True)

    checklist: list[dict[str, Any]] = []
    # Backward-compatible fallback for direct unit test inputs that only provide period totals.
    for a, r in [
        ("annualized_da_pnl_eur", "da_pnl_eur"),
        ("annualized_afrr_pnl_eur", "afrr_pnl_eur"),
        ("annualized_bcm_pnl_eur", "bcm_pnl_eur"),
        ("annualized_bem_pnl_eur", "bem_pnl_eur"),
        ("annualized_id_recourse_pnl_eur", "id_recourse_pnl_eur"),
        ("annualized_degradation_cost_eur", "degradation_cost_eur"),
        ("annualized_penalty_cost_eur", "penalty_cost_eur"),
        ("annualized_aux_cost_eur", "aux_cost_eur"),
    ]:
        if a not in metrics.columns and r in metrics.columns:
            metrics[a] = pd.to_numeric(metrics[r], errors="coerce")
    if "annualized_id_recourse_cost_eur" not in metrics.columns and "annualized_id_recourse_pnl_eur" in metrics.columns:
        metrics["annualized_id_recourse_cost_eur"] = pd.to_numeric(metrics["annualized_id_recourse_pnl_eur"], errors="coerce").fillna(0.0).clip(upper=0.0).abs()

    _plot_quantile_sensitivity(metrics, figures_dir, data_dir, checklist)

    # Tables & decompositions per category
    categories = ["point", "symmetric_interval", "asymmetric_interval"]
    decomp_data: dict[str, pd.DataFrame] = {}
    for cat in categories:
        g = metrics[metrics["quantile_category"] == cat].copy()
        decomp_data[cat] = g
    decomp_ylim = _compute_shared_ylim(
        [decomp_data[c] for c in categories],
        [
            "annualized_da_pnl_eur",
            "annualized_bcm_pnl_eur",
            "annualized_bem_pnl_eur",
            "annualized_id_recourse_pnl_eur",
            "annualized_degradation_cost_eur",
            "annualized_penalty_cost_eur",
            "annualized_aux_cost_eur",
        ],
    )
    costs_ylim_pen = _compute_shared_ylim([decomp_data[c] for c in categories], ["annualized_penalty_cost_eur"])
    costs_ylim_id = _compute_shared_ylim([decomp_data[c] for c in categories], ["annualized_id_recourse_cost_eur"])
    costs_ylim_loss = _compute_shared_ylim([decomp_data[c] for c in categories], ["loss_day_share"])
    idp_ylim = _compute_shared_ylim([decomp_data[c] for c in categories], ["annualized_id_recourse_cost_eur", "annualized_penalty_cost_eur"])

    for cat in categories:
        g = decomp_data[cat]
        if g.empty:
            checklist.append({"output": f"quantile_profit_{cat}", "implemented": "yes", "status": "skipped_no_data", "path": "", "notes": "no data"})
            checklist.append({"output": f"detailed_performance_{cat}", "implemented": "yes", "status": "skipped_no_data", "path": "", "notes": "no data"})
            checklist.append({"output": f"revenue_cost_decomposition_{cat}", "implemented": "yes", "status": "skipped_no_data", "path": "", "notes": "no data"})
            checklist.append({"output": f"costs_lossday_{cat}", "implemented": "yes", "status": "skipped_no_data", "path": "", "notes": "no data"})
            checklist.append({"output": f"id_penalty_sensitivity_{cat}", "implemented": "yes", "status": "skipped_no_data", "path": "", "notes": "no data"})
            continue
        pivot = g.pivot_table(index="quantile_policy", columns="model", values="annualized_realized_net_profit_eur", aggfunc="mean")
        pivot["avg"] = pivot.mean(axis=1)
        pivot = pivot.sort_index()
        latex_df_numeric = pivot.reset_index()
        model_cols = [c for c in pivot.columns if c != "avg"]
        latex_df = latex_df_numeric.copy().astype(object)
        for i in latex_df_numeric.index:
            if model_cols:
                row_vals = pd.to_numeric(latex_df_numeric.loc[i, model_cols], errors="coerce")
                if row_vals.notna().any():
                    best_col = row_vals.idxmax()
                    best_val = float(latex_df_numeric.loc[i, best_col])
                    latex_df.loc[i, best_col] = f"\\textbf{{{best_val:.2f}}}"
        avg_row = {"quantile_policy": "AVERAGE"}
        for c in model_cols + ["avg"]:
            avg_row[c] = float(pd.to_numeric(pivot[c], errors="coerce").mean())
        latex_df = pd.concat([latex_df, pd.DataFrame([avg_row]).astype(object)], ignore_index=True)
        tex = _latex_table(latex_df, f"Annualized profit by quantile policy ({cat})", f"tab:quantile_profit_{cat}")
        (tables_dir / f"quantile_profit_{cat}.tex").write_text(tex, encoding="utf-8")
        pd.concat([latex_df_numeric, pd.DataFrame([avg_row])], ignore_index=True).to_csv(
            tables_dir / f"quantile_profit_{cat}.csv", index=False
        )
        checklist.append({"output": f"quantile_profit_{cat}", "implemented": "yes", "status": "generated", "path": str(tables_dir / f"quantile_profit_{cat}.tex"), "notes": ""})

        dcols = [
            "model", "quantile_policy", "annualized_realized_net_profit_eur", "realized_net_profit_eur",
            "annualized_da_pnl_eur", "annualized_afrr_pnl_eur", "annualized_bcm_pnl_eur", "annualized_bem_pnl_eur", "annualized_id_recourse_pnl_eur",
            "annualized_degradation_cost_eur", "annualized_penalty_cost_eur", "cycles", "loss_day_share"
        ]
        det = g[[c for c in dcols if c in g.columns]].copy()
        det.columns = [
            "Model", "Quantile policy", "Annualized net profit", "Multi PnL", "Annualized DA PnL", "Annualized aFRR PnL", "Annualized BCM PnL", "Annualized BEM PnL", "Annualized ID recourse PnL", "Annualized degradation cost", "Annualized penalty cost", "Cycles", "Loss-day share"
        ][:len(det.columns)]
        (tables_dir / f"detailed_performance_{cat}.csv").write_text(det.to_csv(index=False), encoding="utf-8")
        (tables_dir / f"detailed_performance_{cat}.tex").write_text(
            _latex_table(det, f"Detailed performance ({cat})", f"tab:detailed_performance_{cat}"),
            encoding="utf-8",
        )
        checklist.append({"output": f"detailed_performance_{cat}", "implemented": "yes", "status": "generated", "path": str(tables_dir / f"detailed_performance_{cat}.tex"), "notes": ""})

        # stacked decomposition
        parts = [
            "annualized_da_pnl_eur",
            "annualized_bcm_pnl_eur",
            "annualized_bem_pnl_eur",
            "annualized_id_recourse_pnl_eur",
            "annualized_degradation_cost_eur",
            "annualized_penalty_cost_eur",
            "annualized_aux_cost_eur",
        ]
        dg = g.groupby(["quantile_policy", "model"], as_index=False)[parts].sum()
        dg["degradation_cost_plot_eur"] = -dg["annualized_degradation_cost_eur"].abs()
        dg["penalty_cost_plot_eur"] = -dg["annualized_penalty_cost_eur"].abs()
        dg["aux_cost_plot_eur"] = -dg["annualized_aux_cost_eur"].abs()
        dg.to_csv(data_dir / f"revenue_cost_decomposition_{cat}.csv", index=False)
        fig, ax = plt.subplots(figsize=(13, 5))
        xlabels = [f"{qp}\n{m}" for qp, m in zip(dg["quantile_policy"], dg["model"])]
        x = np.arange(len(dg))
        bottom_pos = np.zeros(len(dg)); bottom_neg = np.zeros(len(dg))
        plot_parts = [
            "annualized_da_pnl_eur",
            "annualized_bcm_pnl_eur",
            "annualized_bem_pnl_eur",
            "annualized_id_recourse_pnl_eur",
            "degradation_cost_plot_eur",
            "penalty_cost_plot_eur",
            "aux_cost_plot_eur",
        ]
        for p in plot_parts:
            vals = pd.to_numeric(dg[p], errors="coerce").fillna(0.0).to_numpy()
            pos = np.where(vals > 0, vals, 0.0)
            neg = np.where(vals < 0, vals, 0.0)
            ax.bar(x, pos, bottom=bottom_pos, label=p)
            ax.bar(x, neg, bottom=bottom_neg, label="_nolegend_")
            bottom_pos += pos; bottom_neg += neg
        ax.set_ylim(*decomp_ylim)
        ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=45, ha="right")
        ax.set_title(f"Revenue/cost decomposition ({cat})")
        ax.legend(loc="best", ncol=2)
        fig.tight_layout(); fig.savefig(figures_dir / f"revenue_cost_decomposition_{cat}.png", dpi=180); plt.close(fig)
        checklist.append({"output": f"revenue_cost_decomposition_{cat}", "implemented": "yes", "status": "generated", "path": str(figures_dir / f"revenue_cost_decomposition_{cat}.png"), "notes": ""})

        # costs+lossday
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        cagg = g.groupby(["quantile_policy", "model"], as_index=False).agg(
            annualized_penalty_cost_eur=("annualized_penalty_cost_eur", "mean"),
            annualized_id_recourse_cost_eur=("annualized_id_recourse_cost_eur", "mean"),
            loss_day_share=("loss_day_share", "mean"),
        )
        cagg.to_csv(data_dir / f"costs_lossday_{cat}.csv", index=False)
        order_map = {q: i for i, q in enumerate(DEFAULT_POLICIES)}
        for model, gm in cagg.groupby("model"):
            gm = gm.copy()
            gm["ord"] = gm["quantile_policy"].map(order_map).fillna(999)
            gm = gm.sort_values("ord")
            xx = np.arange(len(gm))
            axes[0].plot(xx, gm["annualized_penalty_cost_eur"], label=model)
            axes[1].plot(xx, gm["annualized_id_recourse_cost_eur"], label=model)
            axes[2].plot(xx, gm["loss_day_share"], label=model)
            axes[2].set_xticks(xx)
            axes[2].set_xticklabels(gm["quantile_policy"], rotation=45, ha="right")
        axes[0].set_ylim(*costs_ylim_pen)
        axes[1].set_ylim(*costs_ylim_id)
        axes[2].set_ylim(*costs_ylim_loss)
        for a in axes:
            a.legend(loc="best")
        fig.suptitle("Annualized Costs and Loss-Day Share")
        fig.tight_layout(); fig.savefig(figures_dir / f"costs_lossday_{cat}.png", dpi=180); plt.close(fig)
        checklist.append({"output": f"costs_lossday_{cat}", "implemented": "yes", "status": "generated", "path": str(figures_dir / f"costs_lossday_{cat}.png"), "notes": ""})

        # id penalty sensitivity
        fig, ax = plt.subplots(figsize=(12, 5))
        sg = g.groupby(["quantile_policy", "model"], as_index=False)[["annualized_id_recourse_cost_eur", "annualized_penalty_cost_eur"]].mean()
        sg.to_csv(data_dir / f"id_penalty_sensitivity_{cat}.csv", index=False)
        xx = np.arange(len(sg));
        ax.bar(xx, sg["annualized_id_recourse_cost_eur"], label="ID recourse cost")
        ax.bar(xx, sg["annualized_penalty_cost_eur"], bottom=sg["annualized_id_recourse_cost_eur"], label="Penalty cost")
        ax.set_ylim(*idp_ylim)
        ax.set_xticks(xx); ax.set_xticklabels([f"{a}\n{b}" for a, b in zip(sg["quantile_policy"], sg["model"])], rotation=45, ha="right")
        ax.legend(); ax.set_title(f"ID recourse / penalty sensitivity ({cat})")
        fig.tight_layout(); fig.savefig(figures_dir / f"id_penalty_sensitivity_{cat}.png", dpi=180); plt.close(fig)
        checklist.append({"output": f"id_penalty_sensitivity_{cat}", "implemented": "yes", "status": "generated", "path": str(figures_dir / f"id_penalty_sensitivity_{cat}.png"), "notes": ""})

    # cumulative pnl by model best policy
    for model, gm in metrics.groupby("model"):
        best = gm.sort_values("annualized_realized_net_profit_eur", ascending=False).iloc[0]
        hh = hourly_by_output.get(str(best["output_dir"]))
        if hh is None or hh.empty:
            continue
        ts = pd.to_datetime(hh["timestamp_utc"], utc=True, errors="coerce")
        pnl = _pick_first_numeric(hh, ["real_pnl_eur", "pnl_eur"], 0.0).cumsum()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(ts, pnl, label=f"{model}:{best['quantile_policy']}")
        if "naive_pnl_eur" in hh.columns:
            ax.plot(ts, pd.to_numeric(hh["naive_pnl_eur"], errors="coerce").fillna(0.0).cumsum(), label="naive")
        if "rolling_perfect_foresight_pnl_eur" in hh.columns:
            ax.plot(ts, pd.to_numeric(hh["rolling_perfect_foresight_pnl_eur"], errors="coerce").fillna(0.0).cumsum(), label="rolling PF")
        ax.legend(); ax.set_title(f"Cumulative P&L ({model})")
        fig.tight_layout(); fig.savefig(figures_dir / f"cumulative_pnl_{model}.png", dpi=180); plt.close(fig)
    # by policy
    cp_rows: list[pd.DataFrame] = []
    fig_pol, ax_pol = plt.subplots(figsize=(12, 5))
    for _, row in metrics.iterrows():
        hh = hourly_by_output.get(str(row.get("output_dir", "")))
        if hh is None or hh.empty:
            continue
        ts = pd.to_datetime(hh["timestamp_utc"], utc=True, errors="coerce")
        pnl = _pick_first_numeric(hh, ["real_pnl_eur", "pnl_eur"], 0.0).cumsum()
        label = f"{row.get('model')}|{row.get('quantile_policy')}"
        ax_pol.plot(ts, pnl, alpha=0.4, label=label)
        cp_rows.append(pd.DataFrame({"timestamp_utc": ts, "cumulative_pnl_eur": pnl, "model": row.get("model"), "quantile_policy": row.get("quantile_policy")}))
    if cp_rows:
        pd.concat(cp_rows, ignore_index=True).to_csv(data_dir / "cumulative_pnl.csv", index=False)
    ax_pol.set_title("Cumulative P&L by policy")
    fig_pol.tight_layout(); fig_pol.savefig(figures_dir / "cumulative_pnl_by_policy.png", dpi=180); plt.close(fig_pol)
    checklist.append({"output": "cumulative_pnl_by_policy", "implemented": "yes", "status": "generated", "path": str(figures_dir / "cumulative_pnl_by_policy.png"), "notes": ""})

    # daily distribution
    if not daily.empty:
        daily = daily.copy()
        daily["model_policy"] = daily["model"].astype(str) + "|" + daily["quantile_policy"].astype(str)
        groups = [pd.to_numeric(x["daily_pnl_eur"], errors="coerce").dropna().to_numpy() for _, x in daily.groupby("model_policy")]
        labels = [k for k, _ in daily.groupby("model_policy")]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.boxplot(groups, tick_labels=labels, showfliers=False)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title("Daily profit distribution")
        fig.tight_layout(); fig.savefig(figures_dir / "daily_profit_distribution.png", dpi=180); plt.close(fig)
        daily.to_csv(data_dir / "daily_profit_distribution.csv", index=False)
        checklist.append({"output": "daily_profit_distribution", "implemented": "yes", "status": "generated", "path": str(figures_dir / "daily_profit_distribution.png"), "notes": ""})

    # accuracy-profit (optional)
    has_pinball = ("mean_pinball_loss" in metrics.columns) and pd.to_numeric(metrics["mean_pinball_loss"], errors="coerce").notna().any()
    if has_pinball:
        metrics[[c for c in ["model", "quantile_policy", "mean_pinball_loss", "annualized_realized_net_profit_eur"] if c in metrics.columns]].to_csv(
            data_dir / "accuracy_profit_data.csv", index=False
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        for model, gm in metrics.groupby("model"):
            ax.scatter(gm["mean_pinball_loss"], gm["annualized_realized_net_profit_eur"], label=model)
        ax.set_xlabel("Mean pinball loss")
        ax.set_ylabel("Annualized net profit")
        ax.legend(); ax.set_title("Forecast accuracy to profit")
        fig.tight_layout(); fig.savefig(figures_dir / "accuracy_profit_scatter.png", dpi=180); plt.close(fig)
    else:
        warnings.append("forecast accuracy metrics unavailable; wrote placeholder")
        pd.DataFrame(columns=["model", "quantile_policy", "mean_pinball_loss", "annualized_realized_net_profit_eur"]).to_csv(
            data_dir / "accuracy_profit_missing_metrics.csv", index=False
        )
    acc_tab = metrics[[c for c in ["model", "quantile_policy", "annualized_realized_net_profit_eur"] if c in metrics.columns]].copy()
    (tables_dir / "accuracy_profit_table.tex").write_text(_latex_table(acc_tab, "Accuracy-profit link", "tab:accuracy_profit"), encoding="utf-8")
    acc_tab.to_csv(tables_dir / "accuracy_profit_table.csv", index=False)
    checklist.append(
        {
            "output": "accuracy_profit_scatter",
            "implemented": "yes",
            "status": "generated" if (figures_dir / "accuracy_profit_scatter.png").exists() else "skipped_missing_optional_metric",
            "path": str(figures_dir / "accuracy_profit_scatter.png") if (figures_dir / "accuracy_profit_scatter.png").exists() else "",
            "notes": "" if (figures_dir / "accuracy_profit_scatter.png").exists() else "mean_pinball_loss missing",
        }
    )

    # risk robustness
    risk_cols = [
        "model", "quantile_policy", "mean_daily_profit_eur", "median_daily_profit_eur", "loss_day_share", "worst_day_eur",
        "worst_week_eur", "daily_pnl_q05_eur", "cvar_5_eur", "max_drawdown_eur", "profit_volatility_eur",
        "equivalent_full_cycles", "id_recourse_event_count"
    ]
    risk = metrics[[c for c in risk_cols if c in metrics.columns]].copy()
    risk.to_csv(data_dir / "risk_robustness_data.csv", index=False)
    risk.to_csv(tables_dir / "risk_robustness.csv", index=False)
    (tables_dir / "risk_robustness.tex").write_text(_latex_table(risk, "Risk and robustness", "tab:risk_robustness"), encoding="utf-8")

    # representative weeks by model best scenario
    for model, gm in metrics.groupby("model"):
        best = gm.sort_values("annualized_realized_net_profit_eur", ascending=False).iloc[0]
        hh = hourly_by_output.get(str(best["output_dir"]))
        if hh is None or hh.empty:
            continue
        # explicit bcm first
        soc = _pick_first_numeric(hh, ["real_soc_mwh", "soc_mwh"], 0.0)
        da = _pick_first_numeric(hh, ["real_da_buy_mwh", "da_buy_mwh"], 0.0) - _pick_first_numeric(hh, ["real_da_sell_mwh", "da_sell_mwh"], 0.0)
        bcm = _pick_first_numeric(hh, ["real_submitted_bcm_capacity_pos_mw", "submitted_bcm_capacity_pos_mw", "real_submitted_afrr_pos_mw"], 0.0) - _pick_first_numeric(
            hh, ["real_submitted_bcm_capacity_neg_mw", "submitted_bcm_capacity_neg_mw", "real_submitted_afrr_neg_mw"], 0.0
        )
        act = _pick_first_numeric(hh, ["real_act_pos_mwh", "act_pos_mwh"], 0.0) - _pick_first_numeric(hh, ["real_act_neg_mwh", "act_neg_mwh"], 0.0)
        ts = pd.to_datetime(hh["timestamp_utc"], utc=True, errors="coerce")
        rep = pd.DataFrame({"timestamp_utc": ts, "soc_mwh": soc, "da_position_mwh": da, "afrr_reservation_net_mw": bcm, "realized_activation_net_mwh": act})
        if len(rep) > 7 * 24:
            v = rep["soc_mwh"].diff().abs().rolling(24, min_periods=6).mean().fillna(0.0)
            idx = int((v - float(v.median())).abs().idxmin())
            lo = max(0, idx - 84); hi = min(len(rep), lo + 168)
            rep = rep.iloc[lo:hi].copy()
        rep.to_csv(rep_data_dir / f"{model}_representative_week.csv", index=False)
        fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
        axes[0].plot(rep["timestamp_utc"], rep["soc_mwh"]); axes[0].set_ylabel("SoC")
        axes[1].plot(rep["timestamp_utc"], rep["da_position_mwh"]); axes[1].set_ylabel("DA")
        axes[2].plot(rep["timestamp_utc"], rep["afrr_reservation_net_mw"]); axes[2].set_ylabel("aFRR")
        axes[3].plot(rep["timestamp_utc"], rep["realized_activation_net_mwh"]); axes[3].set_ylabel("Act")
        axes[3].set_xlabel("UTC")
        fig.tight_layout(); fig.savefig(rep_fig_dir / f"{model}_representative_week.png", dpi=180); plt.close(fig)

    checklist.append({"output": "risk_robustness", "implemented": "yes", "status": "generated", "path": str(tables_dir / "risk_robustness.tex"), "notes": ""})

    n_fig = len(list(figures_dir.rglob("*.png")))
    n_tbl = len(list(tables_dir.rglob("*.tex")))
    return n_fig, n_tbl, checklist


def _build_simulation_commands(args: argparse.Namespace) -> list[dict[str, Any]]:
    models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    policies = [p.strip() for p in str(args.quantile_policies).split(",") if p.strip()]
    missing = [m for m in models if not args.model_manifest_map.get(m)]
    if missing and not bool(getattr(args, "allow_missing_models", False)):
        raise ValueError(
            f"Missing manifests for models: {', '.join(missing)}. "
            "Provide --model-manifest <model>=<path> for each requested model."
        )
    cmds: list[dict[str, Any]] = []
    for model in models:
        manifest = args.model_manifest_map.get(model, "")
        if not manifest:
            continue
        for qp in policies:
            low, high = _parse_qpair(qp)
            qarg = f"{low}-{high}" if low and high else qp
            out_dir = Path(args.simulation_root) / f"{model}_{args.strategy}_{qp}"
            status = "planned"
            has_summary = (out_dir / "backtest_summary.json").exists()
            has_hourly = (out_dir / "backtest_hourly.parquet").exists() or (out_dir / "backtest_hourly.csv").exists()
            if has_summary and has_hourly:
                if args.reuse_existing:
                    status = "reused"
                elif not args.overwrite:
                    raise ValueError(
                        f"Existing outputs found in {out_dir}. "
                        "Use --reuse-existing or --overwrite."
                    )
            cmd = [
                "./.venv/bin/python",
                "scripts/run_battery_backtest.py",
                "--run-manifest", str(manifest),
                "--split", str(args.split),
                "--trading-strategy", str(args.strategy),
                "--quantile-pairs", qarg,
                "--out-dir", str(out_dir),
                "--da-quantile-role", str(args.da_quantile_role),
                "--final-soc-mode", str(args.final_soc_mode),
            ]
            if bool(args.strict_simulation_validity):
                cmd.append("--strict-simulation-validity")
            if bool(args.export_afrr_bin_ev_audit):
                cmd.append("--export-afrr-bin-ev-audit")
            if bool(args.allow_invalid_output):
                cmd.append("--allow-invalid-output")
            if bool(args.overwrite):
                cmd.append("--clean-output")
            if args.start:
                cmd += ["--start", str(args.start)]
            if args.end:
                cmd += ["--end", str(args.end)]
            logs_dir = Path(args.simulation_root) / "logs"
            log_path = logs_dir / f"{model}_{args.strategy}_{qp}.log"
            cmds.append({"model": model, "quantile_policy": qp, "cmd": cmd, "out_dir": str(out_dir), "status": status, "log_path": str(log_path)})
    return cmds


def _run_simulations_if_requested(args: argparse.Namespace, warnings: list[str]) -> dict[str, Any]:
    if not args.run_simulation:
        return {"run_simulation": False, "commands": [], "results": []}
    commands = _build_simulation_commands(args)
    results: list[dict[str, Any]] = []
    planned = [c for c in commands if c.get("status") == "planned"]
    reused = [c for c in commands if c.get("status") == "reused"]
    for r in reused:
        results.append(
            {
                "model": r["model"],
                "quantile_policy": r["quantile_policy"],
                "strategy": args.strategy,
                "status": "reused",
                "exit_code": 0,
                "out_dir": r["out_dir"],
                "cmd": r["cmd"],
                "log_path": r["log_path"],
            }
        )

    def _run_one(rec: dict[str, Any]) -> dict[str, Any]:
        lp = Path(rec["log_path"])
        lp.parent.mkdir(parents=True, exist_ok=True)
        with lp.open("w", encoding="utf-8") as f:
            cp = subprocess.run(rec["cmd"], stdout=f, stderr=subprocess.STDOUT, text=True)
        return {
            "model": rec["model"],
            "quantile_policy": rec["quantile_policy"],
            "strategy": args.strategy,
            "status": "completed" if cp.returncode == 0 else "failed",
            "exit_code": int(cp.returncode),
            "out_dir": rec["out_dir"],
            "cmd": rec["cmd"],
            "log_path": rec["log_path"],
        }

    workers = max(1, int(getattr(args, "workers", 1)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_run_one, rec) for rec in planned]
        for fu in concurrent.futures.as_completed(futures):
            results.append(fu.result())
    results = [_inspect_run_artifacts(r) for r in results]
    failed = [r for r in results if str(r.get("failure_class", "")).startswith("failed") or str(r.get("failure_class", "")).startswith("process_") or str(r.get("failure_class", "")).startswith("strict_")]
    if failed:
        warnings.append(f"{len(failed)} simulation runs failed; partial diagnostics enabled")
    return {"run_simulation": True, "commands": commands, "results": results, "failed_count": len(failed)}


def _parse_model_manifest_map(args: argparse.Namespace) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in args.model_manifest:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _tail_file(path: Path, n_lines: int = 120) -> str:
    if not path.exists():
        return ""
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lines = txt.splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-n_lines:])


def _inspect_run_artifacts(rec: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(str(rec.get("out_dir", "")))
    summary_path = out_dir / "backtest_summary.json"
    hourly_parq = out_dir / "backtest_hourly.parquet"
    hourly_csv = out_dir / "backtest_hourly.csv"
    hourly_path = hourly_parq if hourly_parq.exists() else (hourly_csv if hourly_csv.exists() else None)
    summary_exists = summary_path.exists()
    hourly_exists = hourly_path is not None
    artifacts_exist = bool(summary_exists or hourly_exists)
    simulation_valid = np.nan
    thesis_reportable = np.nan
    invalid_reason = ""
    if summary_exists:
        try:
            sm = _load_json(summary_path)
            simulation_valid = _safe_float(sm.get("simulation_valid", np.nan))
            thesis_reportable = _safe_float(sm.get("thesis_reportable", np.nan))
            invalid_reason = str(sm.get("invalid_reason", "") or "")
        except Exception:
            pass

    status = str(rec.get("status", "unknown"))
    exit_code = int(rec.get("exit_code", 1))
    if status == "reused":
        fclass = "reused"
    elif exit_code == 0 and np.isfinite(thesis_reportable) and thesis_reportable >= 0.5:
        fclass = "completed_reportable"
    elif exit_code == 0 and artifacts_exist:
        fclass = "completed_nonreportable"
    elif exit_code != 0 and summary_exists and np.isfinite(thesis_reportable) and thesis_reportable < 0.5:
        fclass = "strict_invalid_with_artifacts"
    elif exit_code != 0 and artifacts_exist:
        fclass = "failed_with_artifacts"
    elif exit_code < 0:
        fclass = "process_crash"
    else:
        fclass = "failed_without_artifacts"

    log_path = Path(str(rec.get("log_path", "")))
    log_tail = _tail_file(log_path, n_lines=120)
    exc_type = ""
    exc_msg = ""
    tb_tail = ""
    suspected_stage = "unknown"
    if log_tail:
        lines = log_tail.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            ln = lines[i].strip()
            if "Traceback (most recent call last):" in ln:
                tb = lines[i:]
                tb_tail = "\n".join(tb)
                for ln2 in reversed(tb):
                    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", ln2.strip())
                    if m:
                        exc_type, exc_msg = m.group(1), m.group(2)
                        break
                break
        lt = log_tail.lower()
        if "run_battery_backtest.py" in lt:
            suspected_stage = "run_battery_backtest"
        elif "strict validity" in lt:
            suspected_stage = "strict_validity_output_guard"
        elif "forecast warehouse" in lt:
            suspected_stage = "forecast_loading_or_early_run_stage"

    return {
        **rec,
        "strategy": str(rec.get("strategy", "multi")),
        "summary_path": str(summary_path),
        "hourly_path": "" if hourly_path is None else str(hourly_path),
        "artifacts_exist": bool(artifacts_exist),
        "summary_exists": bool(summary_exists),
        "hourly_exists": bool(hourly_exists),
        "simulation_valid": simulation_valid,
        "thesis_reportable": thesis_reportable,
        "invalid_reason": invalid_reason,
        "failure_class": fclass,
        "log_tail": log_tail,
        "exception_type": exc_type,
        "exception_message": exc_msg,
        "traceback_tail": tb_tail,
        "suspected_failure_stage": suspected_stage,
    }


def _write_failed_reports(out_dir: Path, failed: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    csv_path = out_dir / "failed_runs.csv"
    json_path = out_dir / "failed_runs.json"
    md_path = out_dir / "failed_runs.md"
    pd.DataFrame(failed).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Failed RQ3 Simulation Runs", ""]
    for fr in failed:
        lines += [
            f"## {fr.get('model','')} | {fr.get('strategy','')} | {fr.get('quantile_policy','')}",
            "",
            f"- Exit code: {fr.get('exit_code')}",
            f"- Failure class: {fr.get('failure_class')}",
            f"- Invalid reason: {fr.get('invalid_reason','')}",
            f"- Output dir: {fr.get('out_dir','')}",
            f"- Log path: {fr.get('log_path','')}",
            "- Command:",
            "```bash",
            " ".join(str(x) for x in fr.get("cmd", [])),
            "```",
            "",
            "### Log tail",
            "```text",
            str(fr.get("log_tail", "")),
            "```",
            "",
            "### Suggested rerun",
            "```bash",
            " ".join(str(x) for x in fr.get("cmd", [])),
            "```",
            "",
        ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, json_path, md_path


def _discover_artifacts_from_run_results(run_meta: dict[str, Any], warnings: list[str], include_invalid: bool) -> SimulationArtifacts:
    recs: list[dict[str, Any]] = []
    for r in run_meta.get("results", []):
        fc = str(r.get("failure_class", ""))
        status = str(r.get("status", ""))
        if status == "reused" or fc.startswith("completed"):
            pass
        elif include_invalid and (fc.startswith("strict_invalid") or fc.startswith("failed_with_artifacts")):
            pass
        else:
            continue
        sp = Path(str(r.get("summary_path", "")))
        hp = Path(str(r.get("hourly_path", "")))
        if not sp.exists() or not hp.exists():
            continue
        ql, qh = _parse_qpair(str(r.get("quantile_policy", "")))
        recs.append(
            {
                "source": "current_run_results",
                "output_dir": str(Path(str(r.get("out_dir", ""))).resolve()),
                "summary_path": str(sp),
                "hourly_path": str(hp),
                "scenario": str(r.get("quantile_policy", Path(str(r.get("out_dir", ""))).name)),
                "quantile_low": ql,
                "quantile_high": qh,
                "model": str(r.get("model", _infer_model_label(str(r.get("out_dir", ""))))),
            }
        )
    return SimulationArtifacts(metrics_rows=[], daily_rows=[], scenario_records=recs, warnings=warnings)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="RQ3 strategy diagnostics (reads existing simulation outputs only)")
    ap.add_argument("--simulation-root", default="artifacts/simulation_runs/rq3_test_all_models_quantiles_multi")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--figures-dir", default=None)
    ap.add_argument("--tables-dir", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--scenario-dir", action="append", default=[], help="Repeatable explicit scenario output directory override.")
    ap.add_argument("--skip-simulation", action="store_true")
    ap.add_argument("--run-simulation", action="store_true")
    ap.add_argument("--strategy", default="multi")
    ap.add_argument("--include-invalid", action="store_true")
    ap.add_argument("--list-scenarios", action="store_true")
    ap.add_argument("--models", default=None)
    ap.add_argument("--model-manifest", action="append", default=[])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-partial-results", action="store_true")
    ap.add_argument("--allow-missing-models", action="store_true")
    ap.add_argument("--da-quantile-role", default=None)
    ap.add_argument("--final-soc-mode", default=None)
    ap.add_argument("--export-afrr-bin-ev-audit", action="store_true")
    ap.add_argument("--allow-invalid-output", action="store_true")
    ap.add_argument("--strict-simulation-validity", dest="strict_simulation_validity", action="store_true")
    ap.add_argument("--no-strict-simulation-validity", dest="strict_simulation_validity", action="store_false")
    ap.add_argument("--quantile-policies", default=None)
    ap.add_argument("--split", default=None)
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    if args.run_simulation:
        raise ValueError(
            "generate_strategy_diagnostics.py no longer runs simulations. "
            "Run scripts/run_battery_backtest.py first, then call this script with --simulation-root."
        )
    deprecated_flags = {
        "--models": args.models,
        "--model-manifest": args.model_manifest,
        "--workers": args.workers,
        "--reuse-existing": args.reuse_existing,
        "--overwrite": args.overwrite,
        "--allow-partial-results": args.allow_partial_results,
        "--allow-missing-models": args.allow_missing_models,
        "--da-quantile-role": args.da_quantile_role,
        "--final-soc-mode": args.final_soc_mode,
        "--export-afrr-bin-ev-audit": args.export_afrr_bin_ev_audit,
        "--allow-invalid-output": args.allow_invalid_output,
        "--strict-simulation-validity": args.strict_simulation_validity,
        "--quantile-policies": args.quantile_policies,
        "--split": args.split,
    }
    used_deprecated = [flag for flag, value in deprecated_flags.items() if value not in (None, False, [], "")]
    if used_deprecated:
        raise ValueError(
            "generate_strategy_diagnostics.py is diagnostics-only. Unsupported options: "
            + ", ".join(used_deprecated)
            + ". Run scripts/run_battery_backtest.py first, then call this script with --simulation-root."
        )

    sim_root = Path(args.simulation_root)
    out_dir = Path(args.out_dir) if args.out_dir else sim_root
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    failed_runs_artifacts: dict[str, str] = {}
    explicit_scenario_dirs = [Path(p) for p in args.scenario_dir]
    art = discover_simulation_artifacts(
        sim_root,
        scenario_dirs=explicit_scenario_dirs or None,
        include_invalid=bool(args.include_invalid),
    )
    scenario_source = "explicit_scenario_dirs" if explicit_scenario_dirs else "recursive_discovery"
    stale_excluded = 0
    warnings.extend(art.warnings)
    if args.skip_simulation:
        warnings.append("--skip-simulation is deprecated no-op; diagnostics always reads existing results.")

    start = pd.to_datetime(args.start, utc=True, errors="coerce") if args.start else None
    end = pd.to_datetime(args.end, utc=True, errors="coerce") if args.end else None
    if (args.start and start is pd.NaT) or (args.end and end is pd.NaT):
        raise ValueError("invalid --start/--end timestamp")

    discovery_df = pd.DataFrame(art.scenario_records)
    data_dir = Path(args.data_dir) if args.data_dir else (out_dir / "data")
    figures_dir = Path(args.figures_dir) if args.figures_dir else (out_dir / "figures")
    tables_dir = Path(args.tables_dir) if args.tables_dir else (out_dir / "tables")
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    discovery_df.to_csv(data_dir / "discovered_scenarios.csv", index=False)
    if args.list_scenarios:
        cols = [
            "model",
            "trading_strategy",
            "quantile_policy",
            "simulation_valid",
            "thesis_reportable",
            "invalid_reason",
            "summary_path",
            "hourly_path",
        ]
        print(discovery_df[[c for c in cols if c in discovery_df.columns]].to_string(index=False))
        return

    metrics, daily, hourly_by_output, w2 = _compute_scenario_metrics(art, start, end)
    warnings.extend(w2)
    metrics = _normalize_metrics(metrics, warnings)
    metrics = metrics.loc[metrics["trading_strategy"].astype(str).str.lower().eq(str(args.strategy).lower())].copy()
    if not args.include_invalid and {"thesis_reportable", "simulation_valid"}.issubset(metrics.columns):
        before = len(metrics)
        metrics = metrics.loc[
            (pd.to_numeric(metrics["thesis_reportable"], errors="coerce").fillna(0.0) >= 0.5)
            & (pd.to_numeric(metrics["simulation_valid"], errors="coerce").fillna(0.0) >= 0.5)
        ].copy()
        dropped = before - len(metrics)
        if dropped > 0:
            warnings.append(f"excluded invalid or non-reportable scenarios from figures/tables: {dropped}")
    if metrics.empty:
        raise ValueError("No metrics left after strategy/reportable filtering.")
    if not daily.empty and not metrics.empty:
        keep = metrics[["model", "quantile_policy"]].drop_duplicates()
        daily = daily.merge(keep, on=["model", "quantile_policy"], how="inner")

    metrics.to_csv(data_dir / "scenario_diagnostics.csv", index=False)
    daily.to_csv(data_dir / "daily_scenario_diagnostics.csv", index=False)

    comp = _comparative_diagnosis(metrics, warnings)
    comp.to_csv(data_dir / "comparative_strategic_diagnosis.csv", index=False)

    n_fig, n_tbl, checklist = _render_tables_and_plots(
        metrics,
        daily,
        hourly_by_output,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        data_dir=data_dir,
        warnings=warnings,
    )
    checklist_df = pd.DataFrame(checklist)
    if not checklist_df.empty:
        if "output_name" not in checklist_df.columns and "output" in checklist_df.columns:
            checklist_df["output_name"] = checklist_df["output"]
        if "path" not in checklist_df.columns and "file_path" in checklist_df.columns:
            checklist_df["path"] = checklist_df["file_path"]
        if "status" not in checklist_df.columns:
            checklist_df["status"] = "generated"
    # Ensure checklist completeness.
    existing_outputs = set(checklist_df.get("output_name", pd.Series(dtype=str)).astype(str).tolist()) if not checklist_df.empty else set()
    add_rows = []
    for exp in EXPECTED_OUTPUTS:
        key = exp.replace(".png", "").replace(".tex", "")
        if key in existing_outputs:
            continue
        p = (figures_dir / exp) if exp.endswith(".png") else (tables_dir / exp)
        if p.exists():
            add_rows.append({"output_name": key, "implemented": "yes", "status": "generated", "path": str(p), "notes": "autodetected"})
        else:
            add_rows.append({"output_name": key, "implemented": "yes", "status": "skipped_no_data", "path": "", "notes": "not generated"})
    if add_rows:
        checklist_df = pd.concat([checklist_df, pd.DataFrame(add_rows)], ignore_index=True)

    checklist_df.to_csv(out_dir / "visualization_checklist.csv", index=False)

    def _rel_or_abs(p: Path) -> str:
        try:
            return str(p.relative_to(out_dir))
        except Exception:
            return str(p)
    generated_figures = sorted([_rel_or_abs(p) for p in figures_dir.rglob("*.png")]) if figures_dir.exists() else []
    generated_tables = sorted([_rel_or_abs(p) for p in tables_dir.rglob("*.tex")]) if tables_dir.exists() else []
    generated_data = sorted([_rel_or_abs(p) for p in data_dir.rglob("*.csv")]) if data_dir.exists() else []
    manifest = {
        "simulation_root": str(sim_root),
        "out_dir": str(out_dir),
        "figures_dir": str(figures_dir),
        "tables_dir": str(tables_dir),
        "data_dir": str(data_dir),
        "start": None if start is None else str(start),
        "end": None if end is None else str(end),
        "run_simulation": False,
        "scenario_source": scenario_source,
        "stale_artifacts_excluded_count": int(stale_excluded),
        "partial_results": False,
        "failed_run_count": 0,
        "failed_runs_path": "",
        "discovered_scenarios": int(len(art.scenario_records)),
        "models": sorted(set(metrics["model"].astype(str).tolist())),
        "quantile_policies": sorted(set(metrics["quantile_policy"].astype(str).tolist())),
        "n_figures": int(n_fig),
        "n_tables": int(n_tbl),
        "generated_figures": generated_figures,
        "generated_tables": generated_tables,
        "generated_data_files": generated_data,
        "warnings_count": int(len(warnings)),
        **failed_runs_artifacts,
        "failed_outputs": checklist_df.loc[checklist_df["implemented"].astype(str).str.lower().ne("yes")].to_dict(orient="records") if not checklist_df.empty else [],
        "missing_optional_fields": [w for w in warnings if "optional metric missing" in w],
    }
    (out_dir / "diagnostics_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "diagnostics_warnings.json").write_text(json.dumps(warnings, indent=2), encoding="utf-8")

    print(f"[OK] Diagnostics written to {out_dir}")


if __name__ == "__main__":
    main()
