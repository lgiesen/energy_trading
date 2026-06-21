from __future__ import annotations

import math

import pandas as pd
import pytest

from scripts import analyze_regret_drivers as regret


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
