from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import analyze_regret_drivers as regret


def _scenario(folder: str, model: str, daily: pd.DataFrame, *, is_benchmark: bool = False) -> regret.Scenario:
    return regret.Scenario(
        folder=folder,
        model_key=folder.split("_")[0],
        model=model,
        quantile="benchmark" if is_benchmark else "p50",
        strategy="multi",
        scenario_dir=Path("run") / folder / "multi" / "p50_p50",
        is_benchmark=is_benchmark,
        benchmark_name="RHPF" if is_benchmark else None,
        daily=daily,
    )


def test_daily_regret_is_sorted_before_cumulative_columns() -> None:
    model = _scenario(
        "xgb_p50",
        "XGB",
        pd.DataFrame(
            {
                "date_utc": ["2025-05-03", "2025-05-01", "2025-05-02"],
                "net_revenue_eur": [30.0, 10.0, 20.0],
            }
        ),
    )
    benchmark = _scenario(
        "benchmarks_rhpf",
        "RHPF",
        pd.DataFrame(
            {
                "date_utc": ["2025-05-03", "2025-05-01", "2025-05-02"],
                "net_revenue_eur": [35.0, 20.0, 35.0],
            }
        ),
        is_benchmark=True,
    )
    lookup = regret.ColumnLookup()
    out = regret.build_daily_regret([model, benchmark], benchmark="rhpf", lookup=lookup)

    assert out["date"].tolist() == ["2025-05-01", "2025-05-02", "2025-05-03"]
    assert out["model_daily_profit_eur"].tolist() == [10.0, 20.0, 30.0]
    assert out["benchmark_daily_profit_eur"].tolist() == [20.0, 35.0, 35.0]
    assert out["daily_regret_eur"].tolist() == [10.0, 15.0, 5.0]
    assert out["cumulative_model_profit_eur"].tolist() == [10.0, 30.0, 60.0]
    assert out["cumulative_benchmark_profit_eur"].tolist() == [20.0, 55.0, 90.0]
    assert out["cumulative_regret_eur"].tolist() == [10.0, 25.0, 30.0]
    assert lookup.daily_alignment_rows[0]["model_days"] == 3
    assert lookup.daily_alignment_rows[0]["benchmark_days"] == 3
    assert lookup.daily_alignment_rows[0]["merged_days"] == 3
