#!/usr/bin/env python3
"""Audit load-related feature completeness and leakage risk.

Usage:
    ./.venv/bin/python scripts/audit_load_feature_completeness.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


FEATURE_PATH_CANDIDATES = [
    Path("data/features/all_data_features.parquet"),
]
REFINED_PATH = Path("data/processed/all_data_refined.parquet")


def _pick_existing(cols: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _max_const_run(s: pd.Series) -> int:
    """Longest run of equal consecutive non-null values."""
    x = pd.to_numeric(s, errors="coerce")
    is_valid = x.notna()
    if not is_valid.any():
        return 0
    grp = (x != x.shift(1)) | (~is_valid) | (~is_valid.shift(1).fillna(False))
    run_id = grp.cumsum()
    runs = x[is_valid].groupby(run_id[is_valid]).size()
    return int(runs.max()) if not runs.empty else 0


def _max_abs_err(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    diff = (aa - bb).abs()
    if diff.notna().sum() == 0:
        return float("nan")
    return float(diff.max(skipna=True))


def _source_hint(col: str) -> str:
    c = col.lower()
    if "_entsoe" in c:
        return "ENTSO-E API"
    if "smard" in c:
        return "SMARD API/CSV"
    if "residual_load_calc" == c:
        return "Derived in pipeline"
    return "Merged/Derived (check lineage)"


def _status_from_checks(
    missing_pct: float,
    zero_pct: float,
    max_const_run_h: int,
    leak_flag: bool,
    col: str,
) -> str:
    if leak_flag:
        return "LEAK RISK"
    if missing_pct > 5.0:
        return "TOO MANY MISSINGS"
    if "load" in col and zero_pct > 1.0:
        return "TOO MANY MISSINGS"
    if max_const_run_h >= 48:
        return "TOO MANY MISSINGS"
    return "OK"


def run_audit() -> None:
    feature_path = next((p for p in FEATURE_PATH_CANDIDATES if p.exists()), None)
    if feature_path is None:
        raise FileNotFoundError(f"No feature parquet found in: {FEATURE_PATH_CANDIDATES}")
    if not REFINED_PATH.exists():
        raise FileNotFoundError(f"Missing refined parquet: {REFINED_PATH}")

    fe = pl.read_parquet(feature_path).to_pandas()
    gt = pl.read_parquet(REFINED_PATH).to_pandas()

    for name, df in [("features", fe), ("refined", gt)]:
        if "timestamp_utc" not in df.columns:
            raise KeyError(f"{name}: missing timestamp_utc")
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df.sort_values("timestamp_utc", inplace=True)
        df.set_index("timestamp_utc", inplace=True)

    cols = list(fe.columns)

    # 1) Identification of load-relevant columns with flexible naming.
    actual_load_col = _pick_existing(
        cols,
        [
            "load_actual_entsoe",
            "load_actual",
            "actual_load",
            "actual_consumption",
        ],
    )
    forecast_load_col = _pick_existing(
        cols,
        [
            "load_forecast_da_entsoe",
            "load_forecast_da",
            "forecasted_load",
            "load_forecast",
        ],
    )
    residual_col = _pick_existing(
        cols,
        [
            "residual_load_actual",
            "residual_load_calc",
            "residual_load",
            "residual_consumption",
        ],
    )
    pumped_col = _pick_existing(
        cols,
        [
            "hydro_pumped_actual_entsoe",
            "generation_hydro_pumped_storage_mw",
            "pumped_storage_consumption",
            "pumped_storage_capacity",
        ],
    )

    groups = {
        "Actual Load": actual_load_col,
        "Forecasted Load": forecast_load_col,
        "Residual Load": residual_col,
        "Hydro Pumped Storage": pumped_col,
    }

    print("Detected columns:")
    for g, c in groups.items():
        print(f"- {g}: {c}")

    rows: list[dict[str, Any]] = []

    # 2) Quality & completeness checks.
    for group_name, col in groups.items():
        if col is None:
            rows.append(
                {
                    "Group": group_name,
                    "Column Name": None,
                    "Source": None,
                    "Missing %": np.nan,
                    "Min": np.nan,
                    "Max": np.nan,
                    "Mean": np.nan,
                    "Zero %": np.nan,
                    "Max Const Run (h)": np.nan,
                    "Status": "TOO MANY MISSINGS",
                    "Note": "Column not found",
                }
            )
            continue

        s = pd.to_numeric(fe[col], errors="coerce")
        missing_pct = float(s.isna().mean() * 100.0)
        zero_pct = float((s.fillna(np.nan) == 0).mean() * 100.0)
        min_v = float(s.min(skipna=True)) if s.notna().any() else np.nan
        max_v = float(s.max(skipna=True)) if s.notna().any() else np.nan
        mean_v = float(s.mean(skipna=True)) if s.notna().any() else np.nan
        const_run = _max_const_run(s)

        leak_flag = False
        note = ""

        # 3) Causal logic checks using refined (ground-truth pre-feature) reference.
        if col in gt.columns:
            if group_name == "Actual Load":
                err_t0 = _max_abs_err(fe[col], gt[col])
                err_t2 = _max_abs_err(fe[col], gt[col].shift(2))
                # For strict PiT lag-layer this should align with T-2 for *_actual_entsoe.
                if np.isfinite(err_t0) and np.isfinite(err_t2) and err_t0 <= err_t2:
                    leak_flag = True
                    note = f"actual aligns closer to T0 than T-2 (err_t0={err_t0:.3g}, err_t2={err_t2:.3g})"
                else:
                    note = f"actual lag check OK (err_t0={err_t0:.3g}, err_t2={err_t2:.3g})"
            elif group_name == "Forecasted Load":
                err_t0 = _max_abs_err(fe[col], gt[col])
                # Forecast should be available at decision time for current hour (T-0).
                if np.isfinite(err_t0) and err_t0 > 1e-9:
                    leak_flag = False
                    note = f"forecast not strict T0 everywhere (err_t0={err_t0:.3g}); check gates/horizon columns"
                else:
                    note = "forecast T0 check OK"

        status = _status_from_checks(
            missing_pct=missing_pct,
            zero_pct=zero_pct,
            max_const_run_h=const_run,
            leak_flag=leak_flag,
            col=col,
        )
        if leak_flag:
            status = "LEAK RISK"

        rows.append(
            {
                "Group": group_name,
                "Column Name": col,
                "Source": _source_hint(col),
                "Missing %": round(missing_pct, 4),
                "Min": min_v,
                "Max": max_v,
                "Mean": mean_v,
                "Zero %": round(zero_pct, 4),
                "Max Const Run (h)": const_run,
                "Status": status,
                "Note": note,
            }
        )

    report = pd.DataFrame(rows)
    print("\nAudit summary table:")
    print(report.to_string(index=False))

    # 4) Residual consistency test if enough columns are available.
    print("\nResidual consistency test:")
    required = {"load_actual_entsoe", "wind_onshore_actual_entsoe", "solar_actual_entsoe"}
    if required.issubset(set(fe.columns)):
        wind_total = pd.to_numeric(fe["wind_onshore_actual_entsoe"], errors="coerce")
        if "wind_offshore_actual_entsoe" in fe.columns:
            wind_total = wind_total + pd.to_numeric(fe["wind_offshore_actual_entsoe"], errors="coerce")
        calc_residual = (
            pd.to_numeric(fe["load_actual_entsoe"], errors="coerce")
            - (wind_total + pd.to_numeric(fe["solar_actual_entsoe"], errors="coerce"))
        )

        candidate_residual = _pick_existing(list(fe.columns), ["residual_load_actual", "residual_load_calc"])
        if candidate_residual is not None:
            ref_res = pd.to_numeric(fe[candidate_residual], errors="coerce")
            mae = _max_abs_err(calc_residual, ref_res)
            corr = float(calc_residual.corr(ref_res)) if calc_residual.notna().sum() > 10 else np.nan
            print(
                f"- Compared Calculated_Residual vs {candidate_residual}: "
                f"MAE={mae:.6g}, corr={corr:.6g}"
            )
        else:
            print("- No residual load column found for comparison.")
    else:
        print(f"- Missing required columns for residual test: {sorted(required - set(fe.columns))}")

    # Optional export for thesis appendix.
    out_path = Path("data/reports/processed_audits/load_feature_completeness_audit.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    run_audit()
