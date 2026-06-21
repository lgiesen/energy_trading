from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from scripts import build_rq2_simulation_visualizations as rq2


def test_parse_scenario_folder_maps_models_and_benchmarks() -> None:
    assert rq2.parse_scenario_folder("linear_p50").model_display == "RLQR"
    assert rq2.parse_scenario_folder("xgb_p10").quantile == "p10"
    assert rq2.parse_scenario_folder("benchmarks_naive").benchmark_name == "Naive"
    assert rq2.parse_scenario_folder("logs") is None


def test_infer_duration_prefers_n_days_and_annualization_formula() -> None:
    days, source = rq2._infer_duration_days(pd.DataFrame({"n_days": [60.0, 62.0, 61.0]}), None)
    assert days == 61.0
    assert "n_days" in source
    assert math.isclose(1000.0 * 365.0 / days, 5983.60655737705)


def test_invalid_rows_are_excluded_from_result_table() -> None:
    summary = pd.DataFrame(
        [
            {"model": "Naive", "quantile": "benchmark", "is_benchmark": True, "annualized_profit_eur_per_year": 10.0, "realized_profit_eur": 1.0, "included_in_result_section": True},
            {"model": "RHPF", "quantile": "benchmark", "is_benchmark": True, "annualized_profit_eur_per_year": 20.0, "realized_profit_eur": 2.0, "included_in_result_section": False},
            {"model": "XGB", "quantile": "p10", "is_benchmark": False, "annualized_profit_eur_per_year": 30.0, "realized_profit_eur": 3.0, "included_in_result_section": False},
            {"model": "TFT", "quantile": "p10", "is_benchmark": False, "annualized_profit_eur_per_year": 40.0, "realized_profit_eur": 4.0, "included_in_result_section": True},
        ]
    )
    table, heatmap, bench = rq2.build_result_tables(summary, ["p10"])
    assert table.loc[0, "Naive"] == 10.0
    assert math.isnan(table.loc[0, "RHPF"])
    assert math.isnan(table.loc[0, "XGB"])
    assert table.loc[0, "TFT"] == 40.0
    assert heatmap["model"].tolist() == ["TFT"]
    assert bench["model"].tolist() == ["Naive"]


def test_primary_latex_table_uses_booktabs(tmp_path: Path) -> None:
    table = pd.DataFrame(
        [
            {
                "quantile": "p10",
                "Naive": 10.0,
                "RHPF": float("nan"),
                "RLQR": float("nan"),
                "XGB": 20.0,
                "TFT": 30.0,
                "best_model": "TFT",
                "best_vs_naive_pct": 200.0,
                "best_vs_rhpf_pct": float("nan"),
            }
        ]
    )
    out = tmp_path / "table.tex"
    rq2.write_primary_table(out, table, 61.0)
    text = out.read_text(encoding="utf-8")
    assert r"\toprule" in text
    assert r"\midrule" in text
    assert r"\bottomrule" in text
    assert r"\label{tab:rq2_annualized_pnl_by_model_quantile}" in text


def test_default_output_root_and_latex_include_path_are_stable() -> None:
    assert rq2.DEFAULT_OUT_ROOT == Path("artifacts/benchmark/rq2_simulation_benchmark")
    rel = rq2._thesis_figure_rel(rq2.DEFAULT_OUT_ROOT, "rq2_net_profit_by_quantile_line.png")
    assert rel == "figures/4-results/rq2_simulation_benchmark/result_section/figures/rq2_net_profit_by_quantile_line.png"


def test_best_quantile_component_data_uses_best_valid_model_rows() -> None:
    summary = pd.DataFrame(
        [
            {
                "model": "XGB",
                "quantile": "p10",
                "is_benchmark": False,
                "included_in_result_section": True,
                "annualized_profit_eur_per_year": 100.0,
                "realized_profit_eur": 10.0,
                "annualization_factor": 2.0,
                "da_net_revenue_eur": 8.0,
                "realized_degradation_cost_eur": 3.0,
            },
            {
                "model": "XGB",
                "quantile": "p90",
                "is_benchmark": False,
                "included_in_result_section": True,
                "annualized_profit_eur_per_year": 120.0,
                "realized_profit_eur": 12.0,
                "annualization_factor": 2.0,
                "da_net_revenue_eur": 9.0,
                "realized_degradation_cost_eur": 4.0,
            },
            {
                "model": "TFT",
                "quantile": "p90",
                "is_benchmark": False,
                "included_in_result_section": False,
                "annualized_profit_eur_per_year": 999.0,
                "realized_profit_eur": 99.0,
                "annualization_factor": 2.0,
                "da_net_revenue_eur": 99.0,
                "realized_degradation_cost_eur": 1.0,
            },
        ]
    )
    out = rq2.build_best_quantile_components(summary)
    assert set(out["model"]) == {"XGB"}
    assert set(out["quantile"]) == {"p90"}
    da = out.loc[out["component"].eq("DA net")].iloc[0]
    cost = out.loc[out["component"].eq("Degradation cost")].iloc[0]
    assert da["annualized_component_value_eur_per_year"] == 18.0
    assert cost["annualized_component_value_eur_per_year"] == -8.0


def test_cumulative_pnl_paths_use_best_valid_quantile(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {"folder": "benchmarks_naive", "model": "Naive", "quantile": "benchmark", "is_benchmark": True, "included_in_result_section": True, "annualized_profit_eur_per_year": 10.0},
            {"folder": "xgb_p10", "model": "XGB", "quantile": "p10", "is_benchmark": False, "included_in_result_section": True, "annualized_profit_eur_per_year": 20.0},
            {"folder": "xgb_p90", "model": "XGB", "quantile": "p90", "is_benchmark": False, "included_in_result_section": False, "annualized_profit_eur_per_year": 99.0},
            {"folder": "tft_p50", "model": "TFT", "quantile": "p50", "is_benchmark": False, "included_in_result_section": True, "annualized_profit_eur_per_year": 30.0},
        ]
    )
    for folder, path_type, values in [
        ("benchmarks_naive", "naive", [1.0, 3.0]),
        ("xgb_p10", "model", [2.0, 5.0]),
        ("xgb_p90", "model", [9.0, 99.0]),
        ("tft_p50", "model", [4.0, 8.0]),
    ]:
        d = tmp_path / folder
        d.mkdir()
        pd.DataFrame(
            {
                "path_type": [path_type, path_type],
                "timestamp_utc": ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"],
                "pnl_eur": values,
                "cum_pnl_eur": values,
                "available": [1.0, 1.0],
            }
        ).to_csv(d / "performance_paths_long.csv", index=False)

    out = rq2.build_cumulative_pnl_paths(summary, run_root=tmp_path)
    assert set(out["series"]) == {"Naive", "XGB p10", "TFT p50"}
    assert "XGB p90" not in set(out["series"])
    assert out.loc[out["series"].eq("XGB p10"), "cum_pnl_eur"].tolist() == [2.0, 5.0]


def test_pinball_net_profit_scatter_data_merges_forecast_accuracy_and_profit(tmp_path: Path) -> None:
    joined = tmp_path / "forecast" / "diagnostics" / "joined_predictions"
    joined.mkdir(parents=True)
    base = pd.DataFrame(
        {
            "target_time_utc": ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"],
            "lead_time_h": [1, 1],
            "y_true": [10.0, 20.0],
            "p10": [8.0, 18.0],
            "p50": [12.0, 19.0],
        }
    )
    for model in ["linear", "xgb", "tft"]:
        base.to_parquet(joined / f"{model}__test__pred_da_price.parquet", index=False)

    summary = pd.DataFrame(
        [
            {
                "model": "XGB",
                "quantile": "p50",
                "is_benchmark": False,
                "realized_profit_eur": 100.0,
                "annualized_profit_eur_per_year": 365.0,
                "simulation_valid": 0.0,
                "thesis_reportable": 0.0,
                "included_in_result_section": False,
                "invalid_reason": "diagnostic",
            }
        ]
    )
    out = rq2.build_pinball_net_profit_scatter_data(
        summary=summary,
        forecast_benchmark_dir=tmp_path / "forecast",
        split="test",
        quantiles=["p10", "p50"],
    )
    row = out.loc[(out["model"].eq("XGB")) & (out["quantile"].eq("p50")) & (out["target"].eq("pred_da_price"))].iloc[0]
    assert row["annualized_profit_eur_per_year"] == 365.0
    assert row["n_obs"] == 2
    assert row["mean_pinball_loss"] == 0.75


def test_expected_scenarios_warn_missing_without_zero_fill(tmp_path: Path) -> None:
    (tmp_path / "benchmarks_naive").mkdir()
    pd.DataFrame(
        [
            {
                "scenario": "p50_p50",
                "split": "test",
                "model_key": "linear",
                "n_days": 10.0,
                "simulation_valid": 1.0,
                "thesis_reportable": 1.0,
                "invalid_reason": "none",
                "naive_total_pnl_eur": 100.0,
            }
        ]
    ).to_csv(tmp_path / "benchmarks_naive" / "performance_metrics_all_scenarios.csv", index=False)
    summary, validity, inventory, warnings, *_ = rq2.collect_summaries(
        tmp_path,
        models=["linear"],
        quantiles=["p10"],
        split="test",
        simulation_days=None,
    )
    assert summary.loc[summary["model"] == "Naive", "realized_profit_eur"].iloc[0] == 100.0
    assert not (summary["realized_profit_eur"] == 0.0).any()
    assert "missing expected folder" in " ".join(warnings["message"].astype(str).tolist())
