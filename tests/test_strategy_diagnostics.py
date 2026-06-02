from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pandas as pd
import pytest

import scripts.generate_strategy_diagnostics as gd


def _write_scenario(dirp: Path, *, scenario: str = "p50_p50") -> None:
    dirp.mkdir(parents=True, exist_ok=True)
    (dirp / "backtest_summary.json").write_text(
        json.dumps({"trading_strategy": "multi", "simulation_valid": 1, "thesis_reportable": 1, "invalid_reason": ""}),
        encoding="utf-8",
    )
    h = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-05-01", periods=48, freq="h", tz="UTC"),
            "real_pnl_eur": [1.0] * 48,
            "real_soc_mwh": [10.0 + (i % 6) for i in range(48)],
            "real_da_buy_mwh": [1.0] * 48,
            "real_da_sell_mwh": [0.5] * 48,
            "real_submitted_bcm_capacity_pos_mw": [2.0] * 48,
            "real_submitted_bcm_capacity_neg_mw": [0.0] * 48,
            "real_act_pos_mwh": [0.2] * 48,
            "real_act_neg_mwh": [0.1] * 48,
            "real_revenue_da_eur": [2.0] * 48,
            "real_cost_da_eur": [1.0] * 48,
            "real_revenue_capacity_eur": [0.5] * 48,
            "real_revenue_activation_eur": [0.3] * 48,
            "real_id_pnl_eur": [0.1] * 48,
            "real_degradation_cost_eur": [0.05] * 48,
            "real_penalty_eur": [0.02] * 48,
            "real_aux_cost_eur": [0.01] * 48,
        }
    )
    h.to_csv(dirp / "backtest_hourly.csv", index=False)


def test_comparative_diagnosis_no_crash_with_constraint_pressure() -> None:
    df = pd.DataFrame(
        {
            "model": ["xgb", "xgb"],
            "quantile_policy": ["p30-p30", "p50-p50"],
            "realized_net_profit_eur": [10.0, 12.0],
            "constraint_pressure_index": [0.1, 0.2],
        }
    )
    w: list[str] = []
    out = gd._comparative_diagnosis(df, w)
    assert not out.empty
    assert "mean_constraint_pressure_index" in out.columns


def test_comparative_diagnosis_missing_constraint_pressure_no_crash() -> None:
    df = pd.DataFrame({"model": ["xgb"], "quantile_policy": ["p50-p50"], "realized_net_profit_eur": [1.0]})
    w: list[str] = []
    out = gd._comparative_diagnosis(df, w)
    assert not out.empty
    assert any("constraint_pressure_index missing" in x for x in w)


def test_missing_required_annualized_profit_raises() -> None:
    with pytest.raises(ValueError, match="missing required metrics columns"):
        gd._normalize_metrics(pd.DataFrame({"model": ["xgb"]}), [])


def test_discovery_quantile_sweep_relative_to_sweep_parent(tmp_path: Path) -> None:
    root = tmp_path / "root"
    sweep_dir = root / "run"
    sc = root / "scenarios" / "xgb_multi_p50-p50"
    _write_scenario(sc)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"scenario": "p50_p50", "quantile_low": "p50", "quantile_high": "p50", "output_dir": "../scenarios/xgb_multi_p50-p50"}]
    ).to_csv(sweep_dir / "quantile_sweep_summary.csv", index=False)
    art = gd.discover_simulation_artifacts(root)
    assert len(art.scenario_records) == 1


def test_discovery_nested_summary_hourly(tmp_path: Path) -> None:
    root = tmp_path / "sim"
    _write_scenario(root / "xgb_multi_p50-p50")
    art = gd.discover_simulation_artifacts(root)
    assert art.scenario_records
    rec = art.scenario_records[0]
    assert rec["model"] == "xgboost"
    assert rec["quantile_policy"] == "p50-p50"


def test_direct_scenario_discovery(tmp_path: Path) -> None:
    root = tmp_path / "sim"
    _write_scenario(root)
    art = gd.discover_simulation_artifacts(root)
    assert len(art.scenario_records) == 1
    assert Path(art.scenario_records[0]["scenario_output_dir"]) == root.resolve()


def test_multiple_scenario_discovery(tmp_path: Path) -> None:
    root = tmp_path / "sim"
    _write_scenario(root / "xgb_multi_p50-p50" / "multi" / "p50_p50")
    _write_scenario(root / "xgb_multi_p50-p70" / "multi" / "p50_p70")
    art = gd.discover_simulation_artifacts(root)
    found = {r["quantile_policy"] for r in art.scenario_records}
    assert {"p50-p50", "p50-p70"}.issubset(found)


def test_output_dir_defaults_to_sim_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = tmp_path / "rq3"
    _write_scenario(sim / "xgb_multi_p50-p50")
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_strategy_diagnostics.py",
            "--simulation-root",
            str(sim),
            "--skip-simulation",
        ],
    )
    gd.main()
    assert (sim / "figures").exists()
    assert (sim / "tables").exists()


def test_list_scenarios_writes_csv_and_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    sim = tmp_path / "rq3"
    _write_scenario(sim / "xgb_multi_p50-p70" / "multi" / "p50_p70")
    monkeypatch.setattr("sys.argv", ["generate_strategy_diagnostics.py", "--simulation-root", str(sim), "--list-scenarios"])
    gd.main()
    out = capsys.readouterr().out
    assert "p50-p70" in out
    assert (sim / "data" / "discovered_scenarios.csv").exists()
    assert not (sim / "figures" / "quantile_sensitivity_point.png").exists()


def test_scenario_dir_override_discovers_explicit_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = tmp_path / "rq3"
    scenario = sim / "nested" / "xgb_multi_p50-p70" / "multi" / "p50_p70"
    _write_scenario(scenario)
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        ["generate_strategy_diagnostics.py", "--simulation-root", str(sim), "--scenario-dir", str(scenario), "--out-dir", str(out)],
    )
    gd.main()
    df = pd.read_csv(out / "data" / "scenario_diagnostics.csv")
    assert "p50-p70" in df["quantile_policy"].astype(str).tolist()


def test_tiny_synthetic_creates_key_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = tmp_path / "rq3"
    _write_scenario(sim / "xgb_multi_p50-p50")
    _write_scenario(sim / "tft_multi_p30-p30")
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_strategy_diagnostics.py",
            "--simulation-root",
            str(sim),
            "--skip-simulation",
        ],
    )
    gd.main()
    assert (sim / "figures" / "quantile_sensitivity_point.png").exists()
    assert (sim / "tables" / "risk_robustness.tex").exists()
    assert (sim / "tables" / "quantile_profit_point.tex").exists()
    assert (sim / "tables" / "detailed_performance_point.tex").exists()
    assert (sim / "data" / "revenue_cost_decomposition_point.csv").exists()
    assert (sim / "visualization_checklist.csv").exists()


def test_checklist_marks_optional_accuracy_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = tmp_path / "rq3"
    _write_scenario(sim / "xgb_multi_p50-p50")
    monkeypatch.setattr("sys.argv", ["generate_strategy_diagnostics.py", "--simulation-root", str(sim), "--skip-simulation"])
    gd.main()
    warnings = json.loads((sim / "diagnostics_warnings.json").read_text(encoding="utf-8"))
    assert any("forecast accuracy metrics unavailable" in w for w in warnings)


def test_existing_results_mode_does_not_run_simulation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = tmp_path / "rq3"
    _write_scenario(sim / "xgb_multi_p50-p50")
    monkeypatch.setattr("sys.argv", ["generate_strategy_diagnostics.py", "--simulation-root", str(sim), "--skip-simulation"])
    gd.main()


def test_run_simulation_flag_raises_migration_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["generate_strategy_diagnostics.py", "--simulation-root", str(tmp_path / "sim"), "--run-simulation"],
    )
    with pytest.raises(ValueError, match="no longer runs simulations"):
        gd.main()


def test_pipeline_manifest_only_raises_clear_no_scenarios_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = tmp_path / "rq3"
    sim.mkdir(parents=True, exist_ok=True)
    (sim / "rq3_pipeline_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["generate_strategy_diagnostics.py", "--simulation-root", str(sim), "--skip-simulation"],
    )
    with pytest.raises(FileNotFoundError, match="No simulation artifacts found under"):
        gd.main()

def test_reportability_filter_default_excludes_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = tmp_path / "rq3"
    _write_scenario(sim / "xgb_multi_p50-p50" / "multi" / "p50_p50")
    invalid_dir = sim / "xgb_multi_p30-p30" / "multi" / "p30_p30"
    _write_scenario(invalid_dir)
    (invalid_dir / "backtest_summary.json").write_text(
        json.dumps({"trading_strategy": "multi", "simulation_valid": 0, "thesis_reportable": 0, "invalid_reason": "protected_soc"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["generate_strategy_diagnostics.py", "--simulation-root", str(sim)])
    gd.main()
    df = pd.read_csv(sim / "data" / "scenario_diagnostics.csv")
    assert "p50-p50" in df["quantile_policy"].astype(str).tolist()
    assert "p30-p30" not in df["quantile_policy"].astype(str).tolist()


def test_reportability_filter_include_invalid_includes_both(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = tmp_path / "rq3"
    _write_scenario(sim / "xgb_multi_p50-p50" / "multi" / "p50_p50")
    invalid_dir = sim / "xgb_multi_p30-p30" / "multi" / "p30_p30"
    _write_scenario(invalid_dir)
    (invalid_dir / "backtest_summary.json").write_text(
        json.dumps({"trading_strategy": "multi", "simulation_valid": 0, "thesis_reportable": 0, "invalid_reason": "protected_soc"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["generate_strategy_diagnostics.py", "--simulation-root", str(sim), "--include-invalid"])
    gd.main()
    df = pd.read_csv(sim / "data" / "scenario_diagnostics.csv")
    assert {"p50-p50", "p30-p30"}.issubset(set(df["quantile_policy"].astype(str)))

def test_custom_dirs_honored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = tmp_path / "rq3"
    _write_scenario(sim / "xgb_multi_p50-p50")
    fig = tmp_path / "f"
    tab = tmp_path / "t"
    dat = tmp_path / "d"
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_strategy_diagnostics.py",
            "--simulation-root", str(sim),
            "--skip-simulation",
            "--figures-dir", str(fig),
            "--tables-dir", str(tab),
            "--data-dir", str(dat),
        ],
    )
    gd.main()
    assert fig.exists() and tab.exists() and dat.exists()


def test_compute_shared_ylim_flat_non_degenerate() -> None:
    df = pd.DataFrame({"annualized_realized_net_profit_eur": [1.0, 1.0]})
    lo, hi = gd._compute_shared_ylim([df], ["annualized_realized_net_profit_eur"])
    assert hi > lo


def test_annualization_24_hours_exact_day() -> None:
    ts = pd.Series(pd.date_range("2025-05-01", periods=24, freq="h", tz="UTC"))
    hours, days, fac = gd._infer_analysis_duration_days(ts)
    assert hours == pytest.approx(24.0)
    assert days == pytest.approx(1.0)
    assert fac == pytest.approx(365.0)


def test_decomposition_cost_plot_negative() -> None:
    g = pd.DataFrame(
        {
            "model": ["xgb"],
            "quantile_policy": ["p50-p50"],
            "quantile_category": ["point"],
            "annualized_realized_net_profit_eur": [10.0],
            "annualized_da_pnl_eur": [5.0],
            "annualized_bcm_pnl_eur": [3.0],
            "annualized_bem_pnl_eur": [2.0],
            "annualized_id_recourse_pnl_eur": [-1.0],
            "annualized_id_recourse_cost_eur": [1.0],
            "annualized_degradation_cost_eur": [4.0],
            "annualized_penalty_cost_eur": [2.0],
            "annualized_aux_cost_eur": [1.0],
            "analysis_days": [1.0],
            "output_dir": ["x"],
            "realized_net_profit_eur": [1.0],
            "afrr_pnl_eur": [5.0],
            "cycles": [0.1],
            "loss_day_share": [0.1],
            "equivalent_full_cycles": [0.1],
            "id_recourse_event_count": [1.0],
        }
    )
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        gd._render_tables_and_plots(g, pd.DataFrame(), {}, figures_dir=tmp / "fig", tables_dir=tmp / "tab", data_dir=tmp / "data", warnings=[])
        d = pd.read_csv(tmp / "data" / "revenue_cost_decomposition_point.csv")
        assert d.loc[0, "degradation_cost_plot_eur"] <= 0.0
        assert d.loc[0, "penalty_cost_plot_eur"] <= 0.0
        assert d.loc[0, "aux_cost_plot_eur"] <= 0.0


def test_checklist_completeness_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = tmp_path / "rq3"
    _write_scenario(sim / "xgb_multi_p50-p50")
    monkeypatch.setattr("sys.argv", ["generate_strategy_diagnostics.py", "--simulation-root", str(sim), "--skip-simulation"])
    gd.main()
    ck = pd.read_csv(sim / "visualization_checklist.csv")
    for exp in gd.EXPECTED_OUTPUTS:
        key = exp.replace(".png", "").replace(".tex", "")
        assert key in ck["output_name"].astype(str).values


def test_quantile_categories_mapping() -> None:
    assert gd._categorize_quantile_policy("p50-p50") == "point"
    assert gd._categorize_quantile_policy("p10-p90") == "symmetric_interval"
    assert gd._categorize_quantile_policy("p10-p30") == "asymmetric_interval"


def test_latex_profit_table_has_average_and_bold_best() -> None:
    g = pd.DataFrame(
        {
            "quantile_policy": ["p10-p10", "p50-p50"],
            "model": ["xgboost", "xgboost"],
            "annualized_realized_net_profit_eur": [10.0, 20.0],
            "da_pnl_eur": [1.0, 1.0],
            "bcm_pnl_eur": [1.0, 1.0],
            "bem_pnl_eur": [1.0, 1.0],
            "id_recourse_pnl_eur": [1.0, 1.0],
            "degradation_cost_eur": [-1.0, -1.0],
            "penalty_cost_eur": [-1.0, -1.0],
            "aux_cost_eur": [-1.0, -1.0],
            "afrr_pnl_eur": [2.0, 2.0],
            "cycles": [0.1, 0.2],
            "loss_day_share": [0.0, 0.1],
            "quantile_category": ["point", "point"],
            "output_dir": ["a", "b"],
        }
    )
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        n_fig, n_tbl, _ = gd._render_tables_and_plots(
            g,
            pd.DataFrame(),
            {},
            figures_dir=tmp / "fig",
            tables_dir=tmp / "tab",
            data_dir=tmp / "data",
            warnings=[],
        )
        assert n_tbl >= 1
        tex = (tmp / "tab" / "quantile_profit_point.tex").read_text(encoding="utf-8")
        assert "AVERAGE" in tex
        assert "\\textbf{" in tex


def test_shared_ylim_helper_returns_non_degenerate() -> None:
    d1 = pd.DataFrame({"x": [1.0, 2.0]})
    d2 = pd.DataFrame({"x": [2.0, 3.0]})
    lo, hi = gd._compute_shared_ylim([d1, d2], ["x"])
    assert hi > lo


def test_parse_qpair_folder_names() -> None:
    assert gd._parse_qpair("xgb_multi_p50-p50") == ("p50", "p50")
    assert gd._parse_qpair("xgb_multi_p10_p90") == ("p10", "p90")
    assert gd._parse_qpair("p30-p70") == ("p30", "p70")


def test_traceback_extraction_fields_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = tmp_path / "rq3"
    monkeypatch.setattr("sys.argv", ["generate_strategy_diagnostics.py", "--simulation-root", str(sim), "--run-simulation"])
    with pytest.raises(ValueError, match="no longer runs simulations"):
        gd.main()


def test_empty_root_uses_clear_no_scenarios_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = tmp_path / "rq3"
    sim.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("sys.argv", ["generate_strategy_diagnostics.py", "--simulation-root", str(sim), "--skip-simulation"])
    with pytest.raises(FileNotFoundError) as ei:
        gd.main()
    msg = str(ei.value)
    assert "No simulation artifacts found under" in msg
    assert "--run-simulation" not in msg


def test_costs_lossday_contains_model_dimension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = tmp_path / "rq3"
    _write_scenario(sim / "xgb_multi_p50-p50")
    _write_scenario(sim / "tft_multi_p50-p50")
    monkeypatch.setattr("sys.argv", ["generate_strategy_diagnostics.py", "--simulation-root", str(sim), "--skip-simulation"])
    gd.main()
    df = pd.read_csv(sim / "data" / "costs_lossday_point.csv")
    assert "model" in df.columns
    assert set(df["model"].astype(str)) >= {"xgboost", "tft"}


def test_id_penalty_sensitivity_export_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = tmp_path / "rq3"
    _write_scenario(sim / "xgb_multi_p50-p50")
    monkeypatch.setattr("sys.argv", ["generate_strategy_diagnostics.py", "--simulation-root", str(sim), "--skip-simulation"])
    gd.main()
    df = pd.read_csv(sim / "data" / "id_penalty_sensitivity_point.csv")
    for c in ["model", "quantile_policy", "annualized_id_recourse_cost_eur", "annualized_penalty_cost_eur"]:
        assert c in df.columns
