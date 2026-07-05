from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from scripts import analyze_regret_drivers as regret


def test_default_output_dir_is_under_artifacts() -> None:
    assert regret.DEFAULT_OUT_DIR == Path("artifacts/benchmark/rq2_simulation_benchmark/regret_drivers")


def test_column_discovery_maps_currency_labels() -> None:
    df = pd.DataFrame({"PnL Model (€)": [100.0], "PnL RHPF (€)": [150.0]})
    assert regret.discover_column(df.columns, regret.ROLE_CANDIDATES["realized_profit"], role="realized_profit") == "PnL Model (€)"
    assert regret.discover_column(df.columns, regret.ROLE_CANDIDATES["rhpf_profit"], role="rhpf_profit") == "PnL RHPF (€)"


def test_regret_calculation_uses_benchmark_minus_realized() -> None:
    out = regret.calculate_regret_values(realized_profit=100.0, benchmark_profit=150.0, planned_profit=125.0)
    assert out["total_regret_eur"] == 50.0
    assert math.isclose(out["relative_regret"], 50.0 / 150.0)
    assert out["planning_gap_eur"] == 25.0
    assert math.isclose(out["model_vs_benchmark_pct"], 100.0 / 150.0 * 100.0)


def test_required_column_missing_raises_without_fallback() -> None:
    df = pd.DataFrame({"some_other_column": [1.0]})
    with pytest.raises(regret.MissingRequiredColumnError):
        regret.discover_column(
            df.columns,
            regret.ROLE_CANDIDATES["realized_profit"],
            role="realized_profit",
            source="synthetic",
            required=True,
        )


def test_annualized_regret_table_uses_rhpf_share_definition() -> None:
    model = regret.Scenario(
        folder="xgb_p50",
        model_key="xgb",
        model="XGB",
        quantile="p50",
        strategy="multi",
        scenario_dir=Path("run") / "xgb_p50" / "multi" / "p50_p50",
        is_benchmark=False,
        metrics=pd.DataFrame({"annualized_realized_net_revenue_eur": [80.0]}),
    )
    rhpf = regret.Scenario(
        folder="benchmarks_rhpf",
        model_key="rhpf",
        model="RHPF",
        quantile="benchmark",
        strategy="multi",
        scenario_dir=Path("run") / "benchmarks_rhpf" / "multi" / "p50_p50",
        is_benchmark=True,
        benchmark_name="RHPF",
        metrics=pd.DataFrame({"annualized_realized_net_revenue_eur": [100.0]}),
    )
    out = regret.build_annualized_regret_table([model, rhpf], benchmark="rhpf", lookup=regret.ColumnLookup())
    row = out.iloc[0]
    assert row["Model"] == "XGB"
    assert row["Quantile"] == "p50"
    assert row["Annualized net profit"] == 80.0
    assert row["RHPF annualized profit"] == 100.0
    assert row["Regret vs RHPF"] == 20.0
    assert row["Regret share"] == 20.0
    assert row["Model/RHPF (%)"] == 80.0
