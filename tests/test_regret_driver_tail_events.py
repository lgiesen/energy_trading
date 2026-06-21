from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import analyze_regret_drivers as regret


def _scenario(hourly: pd.DataFrame | None = None, daily: pd.DataFrame | None = None, *, is_benchmark: bool = False) -> regret.Scenario:
    return regret.Scenario(
        folder="benchmarks_rhpf" if is_benchmark else "xgb_p50",
        model_key="rhpf" if is_benchmark else "xgb",
        model="RHPF" if is_benchmark else "XGB",
        quantile="benchmark" if is_benchmark else "p50",
        strategy="multi",
        scenario_dir=Path("run") / ("benchmarks_rhpf" if is_benchmark else "xgb_p50") / "multi" / "p50_p50",
        is_benchmark=is_benchmark,
        benchmark_name="RHPF" if is_benchmark else None,
        hourly=hourly if hourly is not None else pd.DataFrame(),
        daily=daily if daily is not None else pd.DataFrame(),
    )


def _daily_regret() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=10, freq="D").date.astype(str)
    return pd.DataFrame(
        {
            "date": dates,
            "model": "XGB",
            "quantile": "p50",
            "realized_model_profit_eur": range(10),
            "benchmark_profit_eur": range(10, 20),
            "daily_regret_eur": [10.0] * 10,
        }
    )


def test_direction_specific_afrr_tail_columns_are_selected_separately() -> None:
    hourly = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-01-01", periods=10, freq="D"),
            "afrr_capacity_price_pos": range(10),
            "afrr_capacity_price_neg": range(10, 20),
            "afrr_activation_price_pos": range(20, 30),
            "afrr_activation_price_neg": range(30, 40),
            "afrr_activation_rate_pos": [x / 10 for x in range(10)],
            "afrr_activation_rate_neg": [x / 20 for x in range(10)],
        }
    )
    out = regret.build_tail_event_regret(_daily_regret(), [_scenario(hourly)], regret.ColumnLookup())

    def column_for(event_type: str) -> str:
        return str(out.loc[out["tail_event_type"].eq(event_type), "column_used"].iloc[0])

    assert column_for("aFRR capacity price + high tail") == "afrr_capacity_price_pos"
    assert column_for("aFRR capacity price - high tail") == "afrr_capacity_price_neg"
    assert column_for("aFRR activation price + high tail") == "afrr_activation_price_pos"
    assert column_for("aFRR activation price - high tail") == "afrr_activation_price_neg"
    assert column_for("aFRR activation rate + high tail") == "afrr_activation_rate_pos"
    assert column_for("aFRR activation rate - high tail") == "afrr_activation_rate_neg"


def test_missing_negative_tail_column_does_not_fallback_to_positive_column() -> None:
    hourly = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-01-01", periods=10, freq="D"),
            "afrr_capacity_price_pos": range(10),
        }
    )
    lookup = regret.ColumnLookup()
    out = regret.build_tail_event_regret(_daily_regret(), [_scenario(hourly)], lookup)
    neg = out.loc[out["tail_event_type"].eq("aFRR capacity price - high tail")].iloc[0]
    assert neg["data_status"] == "missing_tail_column"
    assert neg["column_used"] == ""
    assert any(row["role"] == "tail:aFRR capacity price - high tail" for row in lookup.missing_rows)


def test_da_high_tail_uses_p95_and_low_tail_uses_p5() -> None:
    hourly = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-01-01", periods=10, freq="D"),
            "target_da_price": range(10),
        }
    )
    out = regret.build_tail_event_regret(_daily_regret(), [_scenario(hourly)], regret.ColumnLookup())
    high = out.loc[out["tail_event_type"].eq("DA price high tail")].iloc[0]
    low = out.loc[out["tail_event_type"].eq("DA price low tail")].iloc[0]
    assert high["tail_side"] == "high"
    assert high["threshold_used"] == 8.549999999999999
    assert high["n_periods"] == 1
    assert low["tail_side"] == "low"
    assert low["threshold_used"] == 0.45
    assert low["n_periods"] == 1


def test_daily_regret_is_sorted_before_cumulative_sum() -> None:
    model = _scenario(
        daily=pd.DataFrame(
            {
                "date_utc": ["2025-01-02", "2025-01-01"],
                "net_revenue_eur": [20.0, 10.0],
            }
        )
    )
    benchmark = _scenario(
        daily=pd.DataFrame(
            {
                "date_utc": ["2025-01-02", "2025-01-01"],
                "net_revenue_eur": [30.0, 15.0],
            }
        ),
        is_benchmark=True,
    )
    out = regret.build_daily_regret([model, benchmark], benchmark="rhpf", lookup=regret.ColumnLookup())
    assert out["date"].tolist() == ["2025-01-01", "2025-01-02"]
    assert out["daily_regret_eur"].tolist() == [5.0, 10.0]
    assert out["cumulative_regret_eur"].tolist() == [5.0, 15.0]
