from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts import validate_simulation_outputs as validate_outputs
from src.energy_trading.simulation.battery_backtest import BatteryBacktester


def _base_summary(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "simulation_valid": 1.0,
        "thesis_reportable": 1.0,
        "invalid_reason": "",
        "fallback_used": 0.0,
        "fallback_mode_counts": "{}",
        "optimization_error_code_counts": '{"ok": 1.0}',
        "simulation_schema_version": "v2",
        "required_summary_fields_version": "v2",
        "code_run_started_at_utc": "2026-01-01T00:00:00Z",
        "command_line_args": "{}",
        "output_was_cleaned": 1.0,
        "infeasible_debug_dump_count": 0.0,
        "accepted_path_infeasible_debug_dump_count": 0.0,
        "candidate_infeasible_debug_dump_count": 0.0,
        "infeasible_debug_dump_paths": [],
        "infeasible_debug_dump_timestamps": [],
        "summary_fields_defaulted": "[]",
        "required_fields_defaulted": "[]",
        "required_fields_computed": "[]",
        "required_fields_missing": "[]",
        "critical_required_fields_defaulted": "[]",
        "optional_fields_defaulted": "[]",
        "required_fields_check_pass": 1.0,
        "precommit_clamp_applied_count": 0.0,
        "activation_split_reconciliation_error_max": 0.0,
        "final_soc_check_pass": 1.0,
        "benchmark_same_rules_gate_consistent": 1.0,
        "global_perfect_foresight_dominance_check_pass": 1.0,
        "global_perfect_foresight_validation_status": "disabled_unverified",
        "id_mode": "none",
        "id_economic_enabled": 0.0,
        "id_technical_repair_enabled": 0.0,
        "total_id_revenue_eur": 0.0,
        "total_id_cost_eur": 0.0,
        "total_id_pnl_eur": 0.0,
        "id_repair_mwh_total": 0.0,
        "id_repair_cost_eur_total": 0.0,
        "id_economic_mwh_total": 0.0,
        "id_economic_pnl_eur_total": 0.0,
        "id_technical_repair_pnl_eur_total": 0.0,
    }
    payload.update(overrides)
    return payload


def test_default_id_mode_mapping() -> None:
    cls = BatteryBacktester
    assert cls.strategy_permissions_from_name("multi").id_mode == "economic"
    assert cls.strategy_permissions_from_name("da_only").id_mode == "none"
    assert cls.strategy_permissions_from_name("afrr_only").id_mode == "technical_repair"
    assert cls.strategy_permissions_from_name("bcm_only").id_mode == "technical_repair"
    assert cls.strategy_permissions_from_name("bem_only").id_mode == "technical_repair"


def test_unknown_id_mode_fails() -> None:
    with pytest.raises(ValueError):
        BatteryBacktester.resolve_strategy_permissions(
            strategy_name="multi",
            allowed_markets=("DA", "aFRR", "ID", "BCM", "BEM"),
            id_mode="bad",
        )


def test_id_recourse_mode_common_allows_id_for_all_strategies() -> None:
    cls = BatteryBacktester
    for strategy in ("multi", "da_only", "afrr_only", "bcm_only", "bem_only"):
        perms = cls.resolve_strategy_permissions(
            strategy_name=strategy,
            allowed_markets=("DA", "aFRR", "ID", "BCM", "BEM"),
            id_recourse_mode="common",
        )
        assert perms.id_mode == "technical_repair"
        assert perms.allow_id


def test_id_recourse_mode_afrr_obligation_only_blocks_da_only() -> None:
    cls = BatteryBacktester
    da_perms = cls.resolve_strategy_permissions(
        strategy_name="da_only",
        allowed_markets=("DA",),
        id_recourse_mode="afrr_obligation_only",
    )
    assert da_perms.id_mode == "none"
    assert not da_perms.allow_id
    afrr_perms = cls.resolve_strategy_permissions(
        strategy_name="afrr_only",
        allowed_markets=("aFRR", "BCM", "BEM"),
        id_recourse_mode="afrr_obligation_only",
    )
    assert afrr_perms.id_mode == "technical_repair"
    assert afrr_perms.allow_id


def test_id_recourse_mode_disabled_blocks_all_id() -> None:
    cls = BatteryBacktester
    for strategy in ("multi", "da_only", "afrr_only", "bcm_only", "bem_only"):
        perms = cls.resolve_strategy_permissions(
            strategy_name=strategy,
            allowed_markets=("DA", "aFRR", "ID", "BCM", "BEM"),
            id_recourse_mode="disabled",
        )
        assert perms.id_mode == "none"
        assert not perms.allow_id


def test_validator_blocks_economic_id_in_baseline(tmp_path: Path) -> None:
    scen = tmp_path / "da_only" / "p50_p50"
    scen.mkdir(parents=True, exist_ok=True)
    payload = _base_summary(
        id_mode="economic",
        id_economic_enabled=1.0,
        total_id_revenue_eur=50.0,
        total_id_cost_eur=20.0,
        total_id_pnl_eur=30.0,
    )
    (scen / "backtest_summary.json").write_text(json.dumps(payload), encoding="utf-8")

    out_csv = tmp_path / "validation.csv"
    out_json = tmp_path / "stats.json"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            sys,
            "argv",
            ["validate_simulation_outputs.py", str(tmp_path), "--out-csv", str(out_csv), "--out-json", str(out_json)],
        )
        validate_outputs.main()

    df = pd.read_csv(out_csv)
    assert float(df["parser_thesis_valid"].iloc[0]) == 0.0


def test_validator_blocks_id_activity_when_recourse_disabled(tmp_path: Path) -> None:
    scen = tmp_path / "multi" / "p50_p50"
    scen.mkdir(parents=True, exist_ok=True)
    payload = _base_summary(
        id_recourse_mode="disabled",
        total_id_revenue_eur=5.0,
        total_id_cost_eur=2.0,
        total_id_pnl_eur=3.0,
    )
    (scen / "backtest_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    out_csv = tmp_path / "validation.csv"
    out_json = tmp_path / "stats.json"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            sys,
            "argv",
            ["validate_simulation_outputs.py", str(tmp_path), "--out-csv", str(out_csv), "--out-json", str(out_json)],
        )
        validate_outputs.main()
    df = pd.read_csv(out_csv)
    assert float(df["parser_thesis_valid"].iloc[0]) == 0.0
