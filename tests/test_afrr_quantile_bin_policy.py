from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_battery_backtest import (
    _apply_quantile_pair_to_warehouse,
    _discover_afrr_bin_ids,
    _expand_quantile_range,
)


def _mk_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "snapshot_time_utc": ["2025-01-01T00:00:00Z"],
            "target_time_utc": ["2025-01-01T01:00:00Z"],
            "lead_time_h": [1],
            "predicted_value": [50.0],
            "p01": [1.0],
            "p05": [5.0],
            "p10": [10.0],
            "p30": [30.0],
            "p50": [50.0],
            "p70": [70.0],
            "p90": [90.0],
            "p95": [95.0],
            "p99": [99.0],
        }
    )


def test_expand_quantile_range_core_cases() -> None:
    assert _expand_quantile_range("p50", "p50") == ["p50"]
    assert _expand_quantile_range("p10", "p30") == ["p10", "p30"]
    assert _expand_quantile_range("p30", "p50") == ["p30", "p50"]
    assert _expand_quantile_range("p50", "p70") == ["p50", "p70"]
    assert _expand_quantile_range("p70", "p90") == ["p70", "p90"]
    assert _expand_quantile_range("p30", "p70") == ["p30", "p50", "p70"]
    assert _expand_quantile_range("p10", "p90") == ["p10", "p30", "p50", "p70", "p90"]
    assert _expand_quantile_range("p05", "p95") == ["p05", "p10", "p30", "p50", "p70", "p90", "p95"]
    assert _expand_quantile_range("p01", "p99") == ["p01", "p05", "p10", "p30", "p50", "p70", "p90", "p95", "p99"]


def test_expand_quantile_range_invalid_cases() -> None:
    try:
        _expand_quantile_range("p70", "p30")
        assert False, "expected ValueError for reversed range"
    except ValueError:
        pass
    try:
        _expand_quantile_range("p20", "p80")
        assert False, "expected ValueError for unknown bins"
    except ValueError:
        pass


def test_apply_quantile_pair_keeps_afrr_predicted_value_and_only_sets_da() -> None:
    warehouse = {
        "pred_da_price": _mk_df(),
        "pred_afrr_capacity_price_pos": _mk_df(),
        "pred_afrr_capacity_price_neg": _mk_df(),
        "pred_afrr_activation_price_pos": _mk_df(),
        "pred_afrr_activation_price_neg": _mk_df(),
        "pred_afrr_activation_rate_pos": _mk_df(),
        "pred_afrr_activation_rate_neg": _mk_df(),
    }
    out = _apply_quantile_pair_to_warehouse(warehouse, q_low="p30", q_high="p70", da_role="mid")

    # DA follows role (mid -> p50)
    assert float(out["pred_da_price"].iloc[0]["predicted_value"]) == 50.0

    # aFRR no longer overwritten with q_high/q_low directional substitution.
    assert float(out["pred_afrr_capacity_price_pos"].iloc[0]["predicted_value"]) == 50.0
    assert float(out["pred_afrr_capacity_price_neg"].iloc[0]["predicted_value"]) == 50.0
    assert float(out["pred_afrr_activation_price_pos"].iloc[0]["predicted_value"]) == 50.0
    assert float(out["pred_afrr_activation_price_neg"].iloc[0]["predicted_value"]) == 50.0


def test_apply_quantile_pair_da_role_isolated() -> None:
    warehouse = {"pred_da_price": _mk_df(), "pred_afrr_capacity_price_pos": _mk_df()}
    out_low = _apply_quantile_pair_to_warehouse(warehouse, q_low="p30", q_high="p70", da_role="low")
    out_high = _apply_quantile_pair_to_warehouse(warehouse, q_low="p30", q_high="p70", da_role="high")

    assert float(out_low["pred_da_price"].iloc[0]["predicted_value"]) == 30.0
    assert float(out_high["pred_da_price"].iloc[0]["predicted_value"]) == 70.0
    assert float(out_low["pred_afrr_capacity_price_pos"].iloc[0]["predicted_value"]) == 50.0
    assert float(out_high["pred_afrr_capacity_price_pos"].iloc[0]["predicted_value"]) == 50.0


def test_discover_afrr_bin_ids_regex() -> None:
    cols = ["reserve_pos_bin_0_mw", "reserve_pos_bin_2_mw", "reserve_pos_bin_1_mw", "x"]
    assert _discover_afrr_bin_ids(cols) == [0, 1, 2]


def test_methodology_docs_consolidated_and_not_outdated() -> None:
    root = Path(__file__).resolve().parents[1]
    main_doc = (root / "docs" / "simulation_methodology.md").read_text(encoding="utf-8")
    rigor_doc = (root / "docs" / "simulation_methodology_and_rigor.md").read_text(encoding="utf-8")
    assert "D-1 09:00 CET" not in main_doc
    assert "D-1 12:00 CET" not in main_doc
    assert "ActRate_pos[t,q]" in main_doc
    assert "ActRate_neg[t,q]" in main_doc
    assert "superseded by docs/simulation_methodology.md" in rigor_doc.lower()
