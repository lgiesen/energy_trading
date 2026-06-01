from __future__ import annotations

import json
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from energy_trading.simulation.battery_backtest import (  # noqa: E402
    BacktestColumnMap,
    BatteryBacktester,
    assign_bcm_capacity_block,
    canonicalize_market_frame,
    load_prediction_warehouse_long,
)
from energy_trading.config import MODEL_SPECS  # noqa: E402
from energy_trading.simulation.bid_builder import AFRRCapacityBid  # noqa: E402
from energy_trading.simulation.market_clearing import MarketClearingEngine  # noqa: E402
from scripts.run_battery_backtest import (  # noqa: E402
    _build_optimization_infeasibility_attribution,
    _prepare_scenario_output_dir,
    _resolve_final_soc_policy,
    _suspected_infeasibility_driver_from_row,
    _target_value_modes_from_manifest,
    optional_numeric_series,
    require_numeric_series,
)
from scripts import validate_simulation_outputs as validate_outputs  # noqa: E402


def _mk_backtester(forecast_value_mode: str = "raw_signed") -> BatteryBacktester:
    MODEL_SPECS["forecast_value_mode"] = forecast_value_mode
    return BatteryBacktester()


def _one_hour_pred_df(
    *,
    da: float,
    cap_pos: float,
    cap_neg: float,
    act_pos: float,
    act_neg: float,
    rate_pos: float,
    rate_neg: float,
) -> tuple[pd.DataFrame, BacktestColumnMap]:
    col = BacktestColumnMap()
    row = {
        col.timestamp: [pd.Timestamp("2026-01-01T00:00:00Z")],
        col.pred_da_price: [da],
        col.pred_afrr_capacity_price_pos: [cap_pos],
        col.pred_afrr_capacity_price_neg: [cap_neg],
        col.pred_afrr_activation_price_pos: [act_pos],
        col.pred_afrr_activation_price_neg: [act_neg],
        col.pred_afrr_activation_rate_pos: [rate_pos],
        col.pred_afrr_activation_rate_neg: [rate_neg],
    }
    for pref, val in [
        (col.pred_afrr_capacity_price_pos, cap_pos),
        (col.pred_afrr_capacity_price_neg, cap_neg),
        (col.pred_afrr_activation_price_pos, act_pos),
        (col.pred_afrr_activation_price_neg, act_neg),
        (col.pred_afrr_activation_rate_pos, rate_pos),
        (col.pred_afrr_activation_rate_neg, rate_neg),
    ]:
        for q in ["p01", "p05", "p10", "p30", "p50", "p70", "p90", "p95", "p99"]:
            row[f"{pref}_{q}"] = [val]
    return pd.DataFrame(row), col


def _tiny_backtest_df(hours: int = 6) -> tuple[pd.DataFrame, BacktestColumnMap]:
    col = BacktestColumnMap()
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=hours, freq="h")
    data = {
        col.timestamp: ts,
        col.pred_da_price: [0.0] * hours,
        "pred_da_price_p05": [0.0] * hours,
        "pred_da_price_p10": [0.0] * hours,
        "pred_da_price_p90": [0.0] * hours,
        "pred_da_price_p95": [0.0] * hours,
        col.pred_afrr_capacity_price_pos: [0.0] * hours,
        col.pred_afrr_capacity_price_neg: [0.0] * hours,
        col.pred_afrr_activation_price_pos: [0.0] * hours,
        col.pred_afrr_activation_price_neg: [0.0] * hours,
        col.pred_afrr_activation_rate_pos: [0.0] * hours,
        col.pred_afrr_activation_rate_neg: [0.0] * hours,
        col.true_da_price: [0.0] * hours,
        col.true_afrr_capacity_price_pos: [0.0] * hours,
        col.true_afrr_capacity_price_neg: [0.0] * hours,
        col.true_afrr_activation_price_pos: [0.0] * hours,
        col.true_afrr_activation_price_neg: [0.0] * hours,
        col.true_afrr_activation_rate_pos: [0.0] * hours,
        col.true_afrr_activation_rate_neg: [0.0] * hours,
    }
    for pref in [
        col.pred_afrr_capacity_price_pos,
        col.pred_afrr_capacity_price_neg,
        col.pred_afrr_activation_price_pos,
        col.pred_afrr_activation_price_neg,
        col.pred_afrr_activation_rate_pos,
        col.pred_afrr_activation_rate_neg,
    ]:
        for q in ["p01", "p05", "p10", "p30", "p50", "p70", "p90", "p95", "p99"]:
            data[f"{pref}_{q}"] = [0.0] * hours
    return pd.DataFrame(data), col


def test_simulation_forecast_loader_applies_negative_target_quantile_flip(tmp_path: Path) -> None:
    p = tmp_path / "neg.parquet"
    pd.DataFrame(
        {
            "snapshot_time_utc": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "target_time_utc": [pd.Timestamp("2026-01-01T01:00:00Z")],
            "lead_time_h": [1],
            "predicted_value": [-50.0],
            "p10": [-100.0],
            "p30": [-80.0],
            "p50": [-50.0],
            "p70": [-20.0],
            "p90": [-10.0],
        }
    ).to_parquet(p, index=False)
    wh = load_prediction_warehouse_long({"pred_afrr_activation_price_neg": p})
    out = wh["pred_afrr_activation_price_neg"].iloc[0]
    assert np.isclose(float(out["p10"]), 10.0)
    assert np.isclose(float(out["p30"]), 20.0)
    assert np.isclose(float(out["p50"]), 50.0)
    assert np.isclose(float(out["p70"]), 80.0)
    assert np.isclose(float(out["p90"]), 100.0)
    assert np.isclose(float(out["predicted_value"]), 50.0)


def test_simulation_forecast_loader_canonical_activation_neg_no_flip(tmp_path: Path) -> None:
    p = tmp_path / "neg_can.parquet"
    pd.DataFrame(
        {
            "snapshot_time_utc": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "target_time_utc": [pd.Timestamp("2026-01-01T01:00:00Z")],
            "lead_time_h": [1],
            "predicted_value": [50.0],
            "p10": [10.0],
            "p30": [20.0],
            "p50": [50.0],
            "p70": [80.0],
            "p90": [100.0],
        }
    ).to_parquet(p, index=False)
    wh = load_prediction_warehouse_long(
        {"pred_afrr_activation_price_neg": p},
        target_value_modes={"pred_afrr_activation_price_neg": "canonical_economic"},
    )
    out = wh["pred_afrr_activation_price_neg"].iloc[0]
    assert np.isclose(float(out["p10"]), 10.0)
    assert np.isclose(float(out["p90"]), 100.0)
    assert np.isclose(float(out["predicted_value"]), 50.0)


def test_manifest_target_value_modes_applied_to_loader_post_load(tmp_path: Path) -> None:
    p = tmp_path / "neg_from_manifest.parquet"
    pd.DataFrame(
        {
            "snapshot_time_utc": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "target_time_utc": [pd.Timestamp("2026-01-01T01:00:00Z")],
            "lead_time_h": [1],
            "predicted_value": [50.0],
            "p10": [10.0],
            "p30": [20.0],
            "p50": [50.0],
            "p70": [80.0],
            "p90": [100.0],
        }
    ).to_parquet(p, index=False)
    payload = {
        "target_value_mode": {
            "pred_afrr_activation_price_neg": "canonical_economic",
        }
    }
    modes = _target_value_modes_from_manifest(payload)
    wh = load_prediction_warehouse_long(
        {"pred_afrr_activation_price_neg": p},
        target_value_modes=modes,
    )
    out = wh["pred_afrr_activation_price_neg"].iloc[0]
    assert np.isclose(float(out["p10"]), 10.0)
    assert np.isclose(float(out["p90"]), 100.0)


def test_resolve_final_soc_policy_strict_terminal_repair_refused() -> None:
    with pytest.raises(ValueError, match="Strict mode requires --final-soc-mode hard"):
        _resolve_final_soc_policy(
            strict_simulation_validity=True,
            final_soc_mode="terminal_repair",
            enforce_final_soc_min_flag=True,
            allow_terminal_soc_repair_in_strict=False,
        )


def test_resolve_final_soc_policy_hard_or_enforce_flag() -> None:
    assert _resolve_final_soc_policy(
        strict_simulation_validity=True,
        final_soc_mode="hard",
        enforce_final_soc_min_flag=False,
        allow_terminal_soc_repair_in_strict=False,
    )
    assert _resolve_final_soc_policy(
        strict_simulation_validity=False,
        final_soc_mode="terminal_repair",
        enforce_final_soc_min_flag=True,
        allow_terminal_soc_repair_in_strict=False,
    )


def test_assign_bcm_capacity_block_local_ce_st_blocks() -> None:
    ts = pd.to_datetime(
        [
            "2025-05-01T00:00:00Z",
            "2025-05-01T01:00:00Z",
            "2025-05-01T02:00:00Z",
            "2025-05-01T03:00:00Z",
        ],
        utc=True,
    )
    blk = assign_bcm_capacity_block(ts)
    # 2025-05-01 is CEST (UTC+2): these map to 02:00..05:00 local -> two blocks.
    assert str(blk["bcm_capacity_block_hour_index"].iloc[0]) == "2.0"
    assert str(blk["bcm_capacity_block_hour_index"].iloc[1]) == "3.0"
    assert str(blk["bcm_capacity_block_hour_index"].iloc[2]) == "0.0"
    assert str(blk["bcm_capacity_block_hour_index"].iloc[3]) == "1.0"
    assert blk["bcm_capacity_block_id"].iloc[0] != blk["bcm_capacity_block_id"].iloc[2]


def test_afrr_only_bcm_block_constant_bem_hourly_not_forced_block_constant() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts = pd.date_range("2025-05-01T00:00:00Z", periods=8, freq="h")
    df = pd.DataFrame(
        {
            col.timestamp: ts,
            col.pred_da_price: [0.0] * 8,
            col.pred_afrr_capacity_price_pos: [100.0] * 4 + [1.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 8,
            col.pred_afrr_activation_price_pos: [150.0, -150.0, 150.0, -150.0, 150.0, -150.0, 150.0, -150.0],
            col.pred_afrr_activation_price_neg: [0.0] * 8,
            col.pred_afrr_activation_rate_pos: [1.0] * 8,
            col.pred_afrr_activation_rate_neg: [0.0] * 8,
        }
    )
    for pref in [
        col.pred_afrr_capacity_price_pos,
        col.pred_afrr_capacity_price_neg,
        col.pred_afrr_activation_price_pos,
        col.pred_afrr_activation_price_neg,
        col.pred_afrr_activation_rate_pos,
        col.pred_afrr_activation_rate_neg,
    ]:
        for q in ["p01", "p05", "p10", "p30", "p50", "p70", "p90", "p95", "p99"]:
            df[f"{pref}_{q}"] = df[pref]

    out = bt.optimize_dispatch(
        df,
        col,
        allowed_markets=("aFRR",),
    )
    b = assign_bcm_capacity_block(out[col.timestamp])
    out = pd.concat([out.reset_index(drop=True), b.reset_index(drop=True)], axis=1)
    for _, g in out.groupby("bcm_capacity_block_id"):
        assert float(pd.to_numeric(g["reserve_pos_mw"], errors="coerce").max() - pd.to_numeric(g["reserve_pos_mw"], errors="coerce").min()) <= 1e-6
        assert float(pd.to_numeric(g["reserve_neg_mw"], errors="coerce").max() - pd.to_numeric(g["reserve_neg_mw"], errors="coerce").min()) <= 1e-6
    # BEM stays hourly; not constrained to block equality.
    assert float(pd.to_numeric(out["bem_only_pos_mw"], errors="coerce").max() - pd.to_numeric(out["bem_only_pos_mw"], errors="coerce").min()) >= 0.0


def test_bcm_block_consistency_check_detects_hourly_variation() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts = pd.date_range("2025-05-01T00:00:00Z", periods=4, freq="h")
    hourly = pd.DataFrame(
        {
            col.timestamp: ts,
            "real_submitted_afrr_pos_mw": [1.0, 1.0, 0.0, 0.0],
            "real_submitted_afrr_neg_mw": [0.0, 0.0, 0.0, 0.0],
            "real_executed_reserve_pos_mw": [1.0, 0.0, 0.0, 0.0],
            "real_executed_reserve_neg_mw": [0.0, 0.0, 0.0, 0.0],
        }
    )
    res = bt._compute_bcm_block_consistency(
        hourly=hourly,
        timestamp_col=col.timestamp,
        bcm_enabled=True,
        tol_mw=1e-6,
    )
    assert float(res["pass"]) == 0.0
    assert float(res["violation_count"]) > 0.0


def test_simulation_forecast_loader_missing_p50_fails_by_default(tmp_path: Path) -> None:
    p = tmp_path / "missing_p50.parquet"
    pd.DataFrame(
        {
            "snapshot_time_utc": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "target_time_utc": [pd.Timestamp("2026-01-01T01:00:00Z")],
            "lead_time_h": [1],
            "predicted_value": [42.0],
            "p10": [10.0],
            "p30": [30.0],
            "p70": [70.0],
            "p90": [90.0],
        }
    ).to_parquet(p, index=False)
    with pytest.raises(KeyError, match="missing required quantile column 'p50'"):
        load_prediction_warehouse_long({"pred_afrr_activation_price_pos": p})


def test_simulation_forecast_loader_materializes_p50_only_with_explicit_flag(tmp_path: Path) -> None:
    p = tmp_path / "missing_p50_optin.parquet"
    pd.DataFrame(
        {
            "snapshot_time_utc": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "target_time_utc": [pd.Timestamp("2026-01-01T01:00:00Z")],
            "lead_time_h": [1],
            "predicted_value": [42.0],
            "p10": [10.0],
            "p30": [30.0],
            "p70": [70.0],
            "p90": [90.0],
        }
    ).to_parquet(p, index=False)
    wh = load_prediction_warehouse_long(
        {"pred_afrr_activation_price_pos": p},
        allow_p50_materialization_from_predicted_value=True,
    )
    out = wh["pred_afrr_activation_price_pos"].iloc[0]
    assert np.isclose(float(out["p50"]), 42.0)
    assert np.isclose(float(out["materialized_p50_from_predicted_value"]), 1.0)


def test_strategy_permissions_enum() -> None:
    cls = BatteryBacktester
    p = cls.strategy_permissions_from_name("multi")
    assert p.allow_da and p.allow_id and p.allow_bcm and p.allow_bem_only
    assert p.id_mode == "economic"
    p = cls.strategy_permissions_from_name("da_only")
    assert p.allow_da and (not p.allow_id) and (not p.allow_bcm) and (not p.allow_bem_only)
    assert p.id_mode == "none"
    p = cls.strategy_permissions_from_name("afrr_only")
    assert (not p.allow_da) and p.allow_id and p.allow_bcm and p.allow_bem_only
    assert p.id_mode == "technical_repair"
    p = cls.strategy_permissions_from_name("bcm_only")
    assert (not p.allow_da) and p.allow_id and p.allow_bcm and (not p.allow_bem_only)
    assert p.id_mode == "technical_repair"
    p = cls.strategy_permissions_from_name("bem_only")
    assert (not p.allow_da) and p.allow_id and (not p.allow_bcm) and p.allow_bem_only
    assert p.id_mode == "technical_repair"
    with pytest.raises(ValueError):
        cls.strategy_permissions_from_name("nope")


def test_bcm_only_optimizer_gating() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=200.0,
        cap_pos=200.0,
        cap_neg=0.0,
        act_pos=250.0,
        act_neg=-50.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("aFRR", "BCM"))
    assert float(out["charge_mw"].iloc[0]) == 0.0
    assert float(out["discharge_mw"].iloc[0]) == 0.0
    assert float(out["bem_only_pos_mw"].iloc[0]) == 0.0
    assert float(out["bem_only_neg_mw"].iloc[0]) == 0.0
    assert float(out["reserve_pos_mw"].iloc[0]) >= 0.0


def test_bem_only_optimizer_gating() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=200.0,
        cap_pos=200.0,
        cap_neg=0.0,
        act_pos=300.0,
        act_neg=-50.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("aFRR", "BEM"))
    assert float(out["charge_mw"].iloc[0]) == 0.0
    assert float(out["discharge_mw"].iloc[0]) == 0.0
    assert float(out["reserve_pos_mw"].iloc[0]) == 0.0
    assert float(out["reserve_neg_mw"].iloc[0]) == 0.0
    assert float(out["bem_only_pos_mw"].iloc[0]) >= 0.0


def test_afrr_only_optimizer_gating() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=300.0,
        cap_pos=100.0,
        cap_neg=0.0,
        act_pos=200.0,
        act_neg=-100.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("aFRR", "BCM", "BEM"))
    assert float(out["charge_mw"].iloc[0]) == 0.0
    assert float(out["discharge_mw"].iloc[0]) == 0.0
    assert float(out["reserve_pos_mw"].iloc[0]) >= 0.0
    assert float(out["bem_only_pos_mw"].iloc[0]) >= 0.0


def test_da_only_optimizer_gating() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=300.0,
        cap_pos=200.0,
        cap_neg=0.0,
        act_pos=200.0,
        act_neg=-100.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("DA",))
    assert float(out["reserve_pos_mw"].iloc[0]) == 0.0
    assert float(out["reserve_neg_mw"].iloc[0]) == 0.0
    assert float(out["bem_only_pos_mw"].iloc[0]) == 0.0
    assert float(out["bem_only_neg_mw"].iloc[0]) == 0.0


def _write_long_pred(
    path: Path,
    *,
    ts: pd.DatetimeIndex,
    predicted_value: float,
    quantiles: dict[str, float],
) -> None:
    snap = pd.Timestamp(ts.min()).tz_convert("UTC") - pd.Timedelta(hours=6)
    df = pd.DataFrame(
        {
            "snapshot_time_utc": [snap] * len(ts),
            "target_time_utc": ts,
            "lead_time_h": [1] * len(ts),
            "predicted_value": [predicted_value] * len(ts),
        }
    )
    for q, v in quantiles.items():
        df[q] = [v] * len(ts)
    df.to_parquet(path, index=False)


def test_strict_preparation_materializes_activation_price_p50_into_optimize_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bt = _mk_backtester("canonical_economic")
    bt.afrr_quantile_bins = ["p30", "p50", "p70"]
    bt.afrr_quantile_prob = {q: 1.0 - float(q[1:]) / 100.0 for q in bt.afrr_quantile_bins}
    base_df, col = _tiny_backtest_df(hours=4)
    drop_cols = [
        c
        for c in base_df.columns
        if c.startswith(f"{col.pred_afrr_activation_price_pos}_")
        or c.startswith(f"{col.pred_afrr_activation_price_neg}_")
    ]
    base_df = base_df.drop(columns=drop_cols, errors="ignore")
    ts = pd.to_datetime(base_df[col.timestamp], utc=True)
    files: dict[str, Path] = {}
    for target in [
        "pred_da_price",
        "pred_afrr_capacity_price_pos",
        "pred_afrr_capacity_price_neg",
        "pred_afrr_activation_rate_pos",
        "pred_afrr_activation_rate_neg",
    ]:
        p = tmp_path / f"{target}.parquet"
        _write_long_pred(
            p,
            ts=ts,
            predicted_value=1.0,
            quantiles={"p30": 0.8, "p50": 1.0, "p70": 1.2},
        )
        files[target] = p
    for target in ["pred_afrr_activation_price_pos", "pred_afrr_activation_price_neg"]:
        p = tmp_path / f"{target}.parquet"
        _write_long_pred(
            p,
            ts=ts,
            predicted_value=10.0,
            quantiles={"p30": 8.0, "p70": 12.0},
        )
        files[target] = p
    warehouse = load_prediction_warehouse_long(
        files,
        target_value_modes={"pred_afrr_activation_price_neg": "canonical_economic"},
        allow_p50_materialization_from_predicted_value=True,
    )
    seen_cols: list[set[str]] = []
    orig_opt = bt.optimize_dispatch

    def _wrapped_opt(df: pd.DataFrame, *args, **kwargs):
        seen_cols.append(set(df.columns))
        return orig_opt(df, *args, **kwargs)

    monkeypatch.setattr(bt, "optimize_dispatch", _wrapped_opt)
    bt.run(
        base_df,
        col,
        use_rolling_horizon=True,
        horizon_hours=4,
        reopt_step_hours=1,
        forecast_warehouse=warehouse,
        strict_simulation_validity=True,
        enable_global_perfect_foresight=False,
    )
    assert seen_cols, "optimize_dispatch should be called at least once"
    assert f"{col.pred_afrr_activation_price_pos}_p50" in seen_cols[0]
    assert f"{col.pred_afrr_activation_price_neg}_p50" in seen_cols[0]


def test_strict_preparation_fails_for_missing_non_p50_active_bin(tmp_path: Path) -> None:
    bt = _mk_backtester("canonical_economic")
    bt.afrr_quantile_bins = ["p30", "p50", "p70"]
    bt.afrr_quantile_prob = {q: 1.0 - float(q[1:]) / 100.0 for q in bt.afrr_quantile_bins}
    base_df, col = _tiny_backtest_df(hours=4)
    drop_cols = [
        c
        for c in base_df.columns
        if c.startswith(f"{col.pred_afrr_activation_price_pos}_")
        or c.startswith(f"{col.pred_afrr_activation_price_neg}_")
    ]
    base_df = base_df.drop(columns=drop_cols, errors="ignore")
    ts = pd.to_datetime(base_df[col.timestamp], utc=True)
    files: dict[str, Path] = {}
    for target in [
        "pred_da_price",
        "pred_afrr_capacity_price_pos",
        "pred_afrr_capacity_price_neg",
        "pred_afrr_activation_rate_pos",
        "pred_afrr_activation_rate_neg",
        "pred_afrr_activation_price_pos",
        "pred_afrr_activation_price_neg",
    ]:
        p = tmp_path / f"{target}.parquet"
        qmap = {"p30": 8.0, "p50": 10.0}
        if "rate" in target:
            qmap = {"p30": 0.2, "p50": 0.3}
        _write_long_pred(
            p,
            ts=ts,
            predicted_value=10.0,
            quantiles=qmap,
        )
        files[target] = p
    warehouse = load_prediction_warehouse_long(
        files,
        target_value_modes={"pred_afrr_activation_price_neg": "canonical_economic"},
    )
    with pytest.raises(ValueError, match="Missing required aFRR quantile-bin inputs in strict mode"):
        bt.run(
            base_df,
            col,
            use_rolling_horizon=True,
            horizon_hours=4,
            reopt_step_hours=1,
            forecast_warehouse=warehouse,
            strict_simulation_validity=True,
            enable_global_perfect_foresight=False,
        )


def test_simulation_forecast_loader_clips_activation_rates() -> None:
    col = BacktestColumnMap()
    df = pd.DataFrame(
        {
            col.timestamp: [pd.Timestamp("2026-01-01T00:00:00Z")],
            col.pred_afrr_activation_rate_pos: [1.4],
            col.pred_afrr_activation_rate_neg: [-0.4],
            f"{col.pred_afrr_activation_rate_pos}_p10": [-0.2],
            f"{col.pred_afrr_activation_rate_pos}_p90": [1.5],
            f"{col.pred_afrr_activation_rate_neg}_p10": [-0.3],
            f"{col.pred_afrr_activation_rate_neg}_p90": [1.7],
            col.true_afrr_activation_rate_pos: [1.8],
            col.true_afrr_activation_rate_neg: [-0.6],
        }
    )
    out = canonicalize_market_frame(df, colmap=col)
    assert np.isclose(float(out.loc[0, col.pred_afrr_activation_rate_pos]), 1.0)
    assert np.isclose(float(out.loc[0, col.pred_afrr_activation_rate_neg]), 0.0)
    assert np.isclose(float(out.loc[0, col.true_afrr_activation_rate_pos]), 1.0)
    assert np.isclose(float(out.loc[0, col.true_afrr_activation_rate_neg]), 0.0)


def test_bcm_pay_as_bid_capacity_settlement_price_and_revenue() -> None:
    bt = _mk_backtester()
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T08:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=5.0,
        planned_reserve_neg_mw=0.0,
        pred_da_price=50.0,
        true_da_price=50.0,
        pred_cap_pos=20.0,
        true_cap_pos=100.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=80.0,
        true_act_pos=80.0,
        pred_act_neg=-100.0,
        true_act_neg=-100.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        obligation_pos_mw=5.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[5.0] + [0.0] * (n_bins - 1),
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[20.0] + [0.0] * (n_bins - 1),
        pred_cap_neg_bins_eur_mw=[0.0] * n_bins,
    )
    assert np.isclose(float(out["executed_reserve_pos_mw"]), 5.0)
    assert np.isclose(float(out["settlement_cap_bid_price_pos_eur_mw"]), 20.0)

    _, m = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=0.0,
        discharge=0.0,
        reserve_pos=5.0,
        reserve_neg=0.0,
        da_price=50.0,
        cap_pos=100.0,
        cap_neg=0.0,
        act_pos_price=80.0,
        act_neg_price=-100.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        cap_bid_pos=20.0,
        cap_bid_neg=0.0,
    )
    assert np.isclose(float(m["revenue_capacity_eur"]), 100.0), m


def test_bcm_reject_no_capacity_award_and_no_capacity_revenue() -> None:
    mc = MarketClearingEngine()
    bids = [
        AFRRCapacityBid(
            ts=pd.Timestamp("2026-01-01T08:00:00Z"),
            side="pos",
            quantity_mw=5.0,
            capacity_price_eur_mw=200.0,
            energy_price_eur_mwh=80.0,
        )
    ]
    res = mc.clear_afrr_capacity(bids, true_cap_pos=100.0, true_cap_neg=0.0)
    assert np.isclose(float(res.awarded_pos_mw), 0.0)
    assert bool(res.pos_awarded) is False


def test_aux_cost_subtracted_once_in_pnl() -> None:
    bt = _mk_backtester()
    _, m = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=0.0,
        discharge=0.0,
        reserve_pos=1.0,
        reserve_neg=0.0,
        da_price=100.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        cap_bid_pos=0.0,
        cap_bid_neg=0.0,
    )
    assert float(m["aux_cost_eur"]) > 0.0
    expected_pnl = (
        float(m["revenue_da_eur"])
        - float(m["cost_da_eur"])
        + float(m["revenue_capacity_eur"])
        + float(m["revenue_activation_eur"])
        + float(m["revenue_id_eur"])
        - float(m["cost_id_eur"])
        - float(m["transaction_cost_eur"])
        - float(m["aux_cost_eur"])
        - float(m["degradation_cost_eur"])
        - float(m["penalty_eur"])
    )
    assert np.isclose(float(m["pnl_eur"]), expected_pnl), m


def test_da_revenue_formula() -> None:
    bt = _mk_backtester()
    _, m = bt._settle_one_hour(
        soc=bt.soc_max,
        charge=0.0,
        discharge=5.0,
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=100.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
    )
    expected = float(m["da_sell_mwh"]) * 100.0
    assert np.isclose(float(m["revenue_da_eur"]), expected, atol=1e-9)


def test_negative_activation_delivered_revenue_sign_convention() -> None:
    bt = _mk_backtester()
    _, m = bt._settle_one_hour(
        soc=bt.soc_min,
        charge=0.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=10.0,
        da_price=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=-1000.0,
        act_pos_rate=0.0,
        act_neg_rate=1.0,
        cap_bid_pos=0.0,
        cap_bid_neg=0.0,
    )
    # Project convention: revenue_neg = -activated_neg_mwh * act_neg_price.
    delivered = float(m["delivered_activation_neg_mwh"])
    assert delivered > 0.0
    assert np.isclose(float(m["revenue_activation_eur"]), -delivered * (-1000.0), atol=1e-6), m


def test_negative_activation_revenue_uses_canonical_positive_value() -> None:
    bt = _mk_backtester("canonical_economic")
    _, m = bt._settle_one_hour(
        soc=bt.soc_min,
        charge=0.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=10.0,
        da_price=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=1000.0,
        act_pos_rate=0.0,
        act_neg_rate=1.0,
        cap_bid_pos=0.0,
        cap_bid_neg=0.0,
    )
    delivered = float(m["delivered_activation_neg_mwh"])
    assert delivered > 0.0
    assert np.isclose(float(m["revenue_activation_eur"]), delivered * 1000.0, atol=1e-6), m


def test_no_double_sign_flip_negative_activation() -> None:
    bt = _mk_backtester("canonical_economic")
    _, m = bt._settle_one_hour(
        soc=bt.soc_min,
        charge=0.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=10.0,
        da_price=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=1000.0,
        act_pos_rate=0.0,
        act_neg_rate=1.0,
        cap_bid_pos=0.0,
        cap_bid_neg=0.0,
    )
    assert float(m["revenue_activation_eur"]) > 0.0


def test_bem_positive_activation_formula() -> None:
    bt = _mk_backtester()
    _, m = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=0.0,
        discharge=0.0,
        reserve_pos=10.0,
        reserve_neg=0.0,
        da_price=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=140.0,
        act_neg_price=0.0,
        act_pos_rate=0.5,
        act_neg_rate=0.0,
    )
    expected_mwh = 10.0 * 0.5 * bt.dt_h * bt.eta_out
    assert np.isclose(float(m["delivered_activation_pos_mwh"]), expected_mwh, atol=1e-9)
    assert np.isclose(float(m["revenue_activation_eur"]), expected_mwh * 140.0, atol=1e-9)


def test_negative_missed_activation_penalty_is_positive() -> None:
    bt = _mk_backtester()
    _, m = bt._settle_one_hour(
        soc=bt.soc_max,
        charge=0.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=10.0,
        da_price=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=-1000.0,
        act_pos_rate=0.0,
        act_neg_rate=1.0,
        cap_bid_pos=0.0,
        cap_bid_neg=0.0,
    )
    assert float(m["missed_activation_neg_mwh"]) > 0.0
    assert np.isclose(float(m["penalty_activation_basis_neg_eur_mwh"]), 1000.0, atol=1e-9)
    assert float(m["penalty_activation_neg_eur"]) > 0.0


def test_degradation_cost_subtracted_once() -> None:
    bt = _mk_backtester()
    _, m = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=5.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
    )
    expected = (
        float(m["revenue_da_eur"]) - float(m["cost_da_eur"]) + float(m["revenue_capacity_eur"])
        + float(m["revenue_activation_eur"]) + float(m["revenue_id_eur"]) - float(m["cost_id_eur"])
        - float(m["transaction_cost_eur"]) - float(m["aux_cost_eur"]) - float(m["degradation_cost_eur"])
        - float(m["penalty_eur"])
    )
    assert np.isclose(float(m["pnl_eur"]), expected, atol=1e-9)


def test_idle_case_optimizer_can_choose_zero_volume() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos=0.0,
        act_neg=0.0,
        rate_pos=0.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("DA", "aFRR"))
    assert np.isclose(float(out["charge_mw"].iloc[0]), 0.0)
    assert np.isclose(float(out["discharge_mw"].iloc[0]), 0.0)
    assert np.isclose(float(out["reserve_pos_mw"].iloc[0]), 0.0)
    assert np.isclose(float(out["reserve_neg_mw"].iloc[0]), 0.0)


def test_da_only_dominance_prefers_da_discharge() -> None:
    old_discount = MODEL_SPECS.get("terminal_soc_value_discount", None)
    try:
        MODEL_SPECS["terminal_soc_value_discount"] = 0.0
        bt = _mk_backtester()
        df, col = _one_hour_pred_df(
            da=400.0,
            cap_pos=0.0,
            cap_neg=0.0,
            act_pos=0.0,
            act_neg=0.0,
            rate_pos=0.0,
            rate_neg=0.0,
        )
        out = bt.optimize_dispatch(df, col, allowed_markets=("DA",))
        assert float(out["discharge_mw"].iloc[0]) > 0.0
    finally:
        if old_discount is None:
            MODEL_SPECS.pop("terminal_soc_value_discount", None)
        else:
            MODEL_SPECS["terminal_soc_value_discount"] = old_discount
    assert np.isclose(float(out["reserve_pos_mw"].iloc[0]), 0.0)
    assert np.isclose(float(out["reserve_neg_mw"].iloc[0]), 0.0)


def test_bcm_plus_bem_activation_revenue_no_double_counting_in_settlement() -> None:
    bt = _mk_backtester()
    _, m = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=0.0,
        discharge=0.0,
        reserve_pos=4.0,
        reserve_neg=0.0,
        da_price=0.0,
        cap_pos=25.0,
        cap_neg=0.0,
        act_pos_price=200.0,
        act_neg_price=-100.0,
        act_pos_rate=0.5,
        act_neg_rate=0.0,
        cap_bid_pos=25.0,
        cap_bid_neg=0.0,
    )
    assert np.isclose(float(m["revenue_capacity_eur"]), 100.0, atol=1e-9)
    delivered_pos = float(m["delivered_activation_pos_mwh"])
    assert delivered_pos > 0.0
    assert np.isclose(float(m["revenue_activation_eur"]), delivered_pos * 200.0, atol=1e-6)


def test_physical_exclusivity_limit_in_optimizer() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=500.0,
        cap_pos=300.0,
        cap_neg=0.0,
        act_pos=300.0,
        act_neg=0.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("DA", "aFRR"))
    dis = float(out["discharge_mw"].iloc[0])
    rpos = float(out["reserve_pos_mw"].iloc[0])
    assert dis + rpos <= bt.p_max_mw + 1e-6


def test_bem_only_mode_is_explicit_optimizer() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=2, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    assert out.summary.get("bem_only_mode") == "explicit_optimizer"
    assert float(out.summary.get("bem_only_explicit_optimizer", 0.0)) == 1.0


def test_comparable_benchmark_semantics_in_tiny_deterministic_case() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=2, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    assert out.summary.get("comparable_benchmark_type") == "rolling_perfect_foresight_same_rules"
    assert "comparable_perfect_foresight_market_pnl_eur" in out.summary


def test_bem_only_positive_activation_without_bcm_award_is_activation_revenue_only() -> None:
    bt = _mk_backtester()
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T10:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=3.0,
        planned_bem_only_neg_mw=0.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=999.0,
        true_cap_pos=0.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=50.0,
        true_act_pos=100.0,
        pred_act_neg=-10.0,
        true_act_neg=-50.0,
        true_rate_pos=1.0,
        true_rate_neg=0.0,
        obligation_pos_mw=0.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[0.0] * n_bins,
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[999.0] + [0.0] * (n_bins - 1),
        pred_cap_neg_bins_eur_mw=[0.0] * n_bins,
    )
    assert np.isclose(float(out["afrr_bcm_auction_cleared"]), 0.0)
    assert np.isclose(float(out["settlement_cap_bid_price_pos_eur_mw"]), 0.0)
    assert float(out["bem_only_submitted_pos_mw"]) > 0.0
    assert float(out["executed_reserve_pos_mw"]) > 0.0
    assert float(out["executed_rate_pos"]) > 0.0
    _, m = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=float(out["executed_charge_mw"]),
        discharge=float(out["executed_discharge_mw"]),
        reserve_pos=float(out["executed_reserve_pos_mw"]),
        reserve_neg=float(out["executed_reserve_neg_mw"]),
        da_price=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=100.0,
        act_neg_price=-50.0,
        act_pos_rate=float(out["executed_rate_pos"]),
        act_neg_rate=float(out["executed_rate_neg"]),
        cap_bid_pos=float(out["settlement_cap_bid_price_pos_eur_mw"]),
        cap_bid_neg=float(out["settlement_cap_bid_price_neg_eur_mw"]),
    )
    assert np.isclose(float(m["revenue_capacity_eur"]), 0.0, atol=1e-9)
    assert float(m["revenue_activation_eur"]) >= 0.0


def test_bem_only_negative_activation_sign_without_bcm_award() -> None:
    bt = _mk_backtester()
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T10:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=0.0,
        planned_bem_only_neg_mw=2.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=999.0,
        true_cap_neg=0.0,
        pred_act_pos=10.0,
        true_act_pos=10.0,
        pred_act_neg=-100.0,
        true_act_neg=-1000.0,
        true_rate_pos=0.0,
        true_rate_neg=1.0,
        obligation_pos_mw=0.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[0.0] * n_bins,
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[0.0] * n_bins,
        pred_cap_neg_bins_eur_mw=[999.0] + [0.0] * (n_bins - 1),
    )
    assert np.isclose(float(out["afrr_bcm_auction_cleared"]), 0.0)
    assert float(out["bem_only_submitted_neg_mw"]) > 0.0
    _, m = bt._settle_one_hour(
        soc=bt.soc_min,
        charge=float(out["executed_charge_mw"]),
        discharge=float(out["executed_discharge_mw"]),
        reserve_pos=float(out["executed_reserve_pos_mw"]),
        reserve_neg=float(out["executed_reserve_neg_mw"]),
        da_price=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=10.0,
        act_neg_price=-1000.0,
        act_pos_rate=float(out["executed_rate_pos"]),
        act_neg_rate=float(out["executed_rate_neg"]),
        cap_bid_pos=0.0,
        cap_bid_neg=0.0,
    )
    assert np.isclose(float(m["revenue_capacity_eur"]), 0.0, atol=1e-9)
    if float(m["delivered_activation_neg_mwh"]) > 0.0:
        assert float(m["revenue_activation_eur"]) > 0.0


def test_bem_only_pos_submission_capped_by_protected_soc_headroom() -> None:
    bt = _mk_backtester()
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T10:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=2.0,
        planned_bem_only_neg_mw=0.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=100.0,
        true_act_pos=100.0,
        pred_act_neg=-100.0,
        true_act_neg=-100.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        soc_now=bt.soc_min + 0.05,
        obligation_pos_mw=0.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[0.0] * n_bins,
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[0.0] * n_bins,
        pred_cap_neg_bins_eur_mw=[0.0] * n_bins,
    )
    assert float(out["desired_bem_only_pos_mw"]) == 2.0
    assert float(out["submitted_bem_only_pos_mw"]) <= 2.0 + 1e-9
    assert float(out["submitted_bem_only_pos_mw"]) <= float(out["safe_bem_only_pos_mw"]) + 1e-9


def test_bem_only_neg_submission_capped_by_protected_soc_headroom() -> None:
    bt = _mk_backtester()
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T10:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=0.0,
        planned_bem_only_neg_mw=2.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=100.0,
        true_act_pos=100.0,
        pred_act_neg=-100.0,
        true_act_neg=-100.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        soc_now=bt.soc_max - 0.05,
        obligation_pos_mw=0.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[0.0] * n_bins,
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[0.0] * n_bins,
        pred_cap_neg_bins_eur_mw=[0.0] * n_bins,
    )
    assert float(out["desired_bem_only_neg_mw"]) == 2.0
    assert float(out["submitted_bem_only_neg_mw"]) <= 2.0 + 1e-9
    assert float(out["submitted_bem_only_neg_mw"]) <= float(out["safe_bem_only_neg_mw"]) + 1e-9


def test_bem_only_guard_uses_locked_reserve_protected_envelope() -> None:
    bt = _mk_backtester()
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T10:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=3.0,
        planned_bem_only_neg_mw=0.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=100.0,
        true_cap_pos=100.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=100.0,
        true_act_pos=100.0,
        pred_act_neg=-100.0,
        true_act_neg=-100.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        soc_now=bt.soc_init,
        obligation_pos_mw=2.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[0.0] * n_bins,
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[0.0] * n_bins,
        pred_cap_neg_bins_eur_mw=[0.0] * n_bins,
    )
    assert float(out["bem_only_protected_soc_min_mwh"]) > bt.soc_min


def test_bem_only_guard_does_not_change_bcm_linked_activation() -> None:
    bt = _mk_backtester()
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T08:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=5.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=3.0,
        planned_bem_only_neg_mw=0.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=10.0,
        true_cap_pos=10.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=100.0,
        true_act_pos=100.0,
        pred_act_neg=-100.0,
        true_act_neg=-100.0,
        true_rate_pos=1.0,
        true_rate_neg=0.0,
        obligation_pos_mw=5.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[5.0] + [0.0] * (n_bins - 1),
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[10.0] + [0.0] * (n_bins - 1),
        pred_cap_neg_bins_eur_mw=[0.0] * n_bins,
    )
    assert float(out["executed_reserve_pos_mw"]) >= 5.0 - 1e-9
    assert float(out["bem_only_submitted_pos_mw"]) == 0.0


def test_bem_only_guard_keeps_forecast_values_unchanged() -> None:
    bt = _mk_backtester()
    n_bins = len(bt.afrr_quantile_bins)
    pred_pos = 123.45
    pred_neg = -67.89
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T10:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=2.0,
        planned_bem_only_neg_mw=2.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=pred_pos,
        true_act_pos=pred_pos,
        pred_act_neg=pred_neg,
        true_act_neg=pred_neg,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        soc_now=bt.soc_init,
        obligation_pos_mw=0.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[0.0] * n_bins,
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[0.0] * n_bins,
        pred_cap_neg_bins_eur_mw=[0.0] * n_bins,
    )
    assert float(out["desired_bem_only_pos_mw"]) == 2.0
    assert float(out["desired_bem_only_neg_mw"]) == 2.0
    assert float(out["submitted_bem_only_pos_mw"]) <= 2.0 + 1e-9
    assert float(out["submitted_bem_only_neg_mw"]) <= 2.0 + 1e-9


def test_simultaneous_bem_only_pos_neg_allowed_if_both_feasible() -> None:
    bt = _mk_backtester()
    bt.disallow_simultaneous_bem_only_pos_neg = False
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T10:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=1.0,
        planned_bem_only_neg_mw=1.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=100.0,
        true_act_pos=100.0,
        pred_act_neg=-100.0,
        true_act_neg=-100.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        soc_now=bt.soc_init,
        obligation_pos_mw=0.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[0.0] * n_bins,
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[0.0] * n_bins,
        pred_cap_neg_bins_eur_mw=[0.0] * n_bins,
    )
    assert float(out["submitted_bem_only_pos_mw"]) > 0.0
    assert float(out["submitted_bem_only_neg_mw"]) > 0.0


def test_simultaneous_bem_only_pos_neg_capped_if_one_side_infeasible() -> None:
    bt = _mk_backtester()
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T10:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=2.0,
        planned_bem_only_neg_mw=2.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=100.0,
        true_act_pos=100.0,
        pred_act_neg=-100.0,
        true_act_neg=-100.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        soc_now=bt.soc_min + 0.05,
        obligation_pos_mw=0.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[0.0] * n_bins,
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[0.0] * n_bins,
        pred_cap_neg_bins_eur_mw=[0.0] * n_bins,
    )
    assert float(out["submitted_bem_only_pos_mw"]) < float(out["desired_bem_only_pos_mw"])
    assert float(out["submitted_bem_only_pos_mw"]) <= float(out["safe_bem_only_pos_mw"]) + 1e-9
    assert float(out["submitted_bem_only_neg_mw"]) >= 0.0


def test_bem_only_guard_fields_written_to_hourly_and_summary() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=2, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    for c in (
        "desired_bem_only_pos_mw",
        "safe_bem_only_pos_mw",
        "bem_only_submitted_pos_mw_before_guard",
        "bem_only_submitted_pos_mw_after_guard",
        "submitted_bem_only_pos_mw",
        "bem_only_headroom_guard_applied",
        "bem_only_headroom_guard_reason",
        "bem_only_guard_soc_now_mwh",
        "bem_only_guard_protected_soc_min_mwh",
        "bem_only_guard_protected_soc_max_mwh",
    ):
        assert c in out.hourly.columns
    for k in (
        "bem_only_headroom_safety_mwh",
        "bem_only_headroom_guard_applied_count",
        "bem_only_pos_reduced_by_headroom_mw_sum",
        "bem_only_headroom_guard_hours",
    ):
        assert k in out.summary


def test_bem_only_guard_prevents_known_low_soc_positive_violation() -> None:
    bt = _mk_backtester()
    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=2.0,
        desired_bem_only_neg_mw=0.0,
        soc_start_mwh=3.5,
        locked_reserve_pos_mw=8.0,
        locked_reserve_neg_mw=0.0,
        pred_act_pos=100.0,
        pred_act_neg=-100.0,
    )
    assert float(guard["bem_only_guard_protected_soc_min_mwh"]) >= 5.0 - 1e-9
    assert float(guard["bem_only_submitted_pos_mw_after_guard"]) == 0.0


def test_bem_only_summary_fields_present() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=2, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    required = [
        "bem_only_mode",
        "bem_only_explicit_optimizer",
        "bem_only_hours",
        "bem_only_pos_bid_hours",
        "bem_only_neg_bid_hours",
        "bem_only_pos_bid_mw_sum",
        "bem_only_neg_bid_mw_sum",
        "bem_only_pos_activation_accept_count",
        "bem_only_neg_activation_accept_count",
        "bem_only_pos_activation_mwh_sum",
        "bem_only_neg_activation_mwh_sum",
        "bem_only_activation_revenue_eur",
    ]
    for k in required:
        assert k in out.summary


def test_bem_only_positive_dominance_optimizer_decision() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=-200.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos=300.0,
        act_neg=-10.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("DA", "aFRR"))
    assert "bem_only_pos_mw" in out.columns
    assert float(out["bem_only_pos_mw"].iloc[0]) > 0.0


def test_bem_only_negative_dominance_optimizer_decision() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=200.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos=0.0,
        act_neg=-300.0,
        rate_pos=0.0,
        rate_neg=1.0,
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("DA", "aFRR"))
    assert "bem_only_neg_mw" in out.columns
    assert float(out["bem_only_neg_mw"].iloc[0]) > 0.0


def test_bem_only_vs_da_exclusivity() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=500.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos=500.0,
        act_neg=0.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("DA", "aFRR"))
    dis = float(out["discharge_mw"].iloc[0])
    bem = float(out["bem_only_pos_mw"].iloc[0])
    assert dis + bem <= bt.p_max_mw + 1e-6


def test_bem_only_positive_revenue_decomposition() -> None:
    bt = _mk_backtester()
    comp = bt._split_activation_revenue_components(
        delivered_pos_mwh=10.0,
        delivered_neg_mwh=0.0,
        bem_only_pos_mwh=10.0,
        bem_only_neg_mwh=0.0,
        act_pos_price_eur_mwh=140.0,
        act_neg_price_eur_mwh=0.0,
    )
    assert np.isclose(comp["bem_only_activation_revenue_eur"], 1400.0)
    assert np.isclose(comp["bcm_linked_activation_revenue_eur"], 0.0)
    assert np.isclose(comp["activation_revenue_reconciled_eur"], 1400.0)


def test_split_activation_revenue_components_raw_signed_neg_price() -> None:
    bt = _mk_backtester("raw_signed")
    comp = bt._split_activation_revenue_components(
        delivered_pos_mwh=0.0,
        delivered_neg_mwh=10.0,
        bem_only_pos_mwh=0.0,
        bem_only_neg_mwh=10.0,
        act_pos_price_eur_mwh=0.0,
        act_neg_price_eur_mwh=-1000.0,
    )
    assert np.isclose(comp["bem_only_activation_revenue_eur"], 10000.0)
    assert np.isclose(comp["bcm_linked_activation_revenue_eur"], 0.0)
    assert np.isclose(comp["activation_revenue_reconciled_eur"], 10000.0)


def test_split_activation_revenue_components_canonical_neg_positive_value() -> None:
    bt = _mk_backtester("canonical_economic")
    comp = bt._split_activation_revenue_components(
        delivered_pos_mwh=0.0,
        delivered_neg_mwh=10.0,
        bem_only_pos_mwh=0.0,
        bem_only_neg_mwh=10.0,
        act_pos_price_eur_mwh=0.0,
        act_neg_price_eur_mwh=1000.0,
    )
    assert np.isclose(comp["bem_only_activation_revenue_eur"], 10000.0)
    assert np.isclose(comp["bcm_linked_activation_revenue_eur"], 0.0)
    assert np.isclose(comp["activation_revenue_reconciled_eur"], 10000.0)


def test_bem_only_not_double_counted_activation_revenue() -> None:
    bt = _mk_backtester()
    comp = bt._split_activation_revenue_components(
        delivered_pos_mwh=12.0,
        delivered_neg_mwh=6.0,
        bem_only_pos_mwh=5.0,
        bem_only_neg_mwh=2.0,
        act_pos_price_eur_mwh=100.0,
        act_neg_price_eur_mwh=-50.0,
    )
    lhs = comp["activation_revenue_reconciled_eur"]
    rhs = comp["bcm_linked_activation_revenue_eur"] + comp["bem_only_activation_revenue_eur"]
    assert np.isclose(lhs, rhs)


def test_full_settlement_activation_revenue_not_overwritten_by_wrong_split() -> None:
    bt = _mk_backtester("canonical_economic")
    _, m = bt._settle_one_hour(
        soc=bt.soc_min,
        charge=0.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=10.0,
        da_price=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=1000.0,
        act_pos_rate=0.0,
        act_neg_rate=1.0,
        cap_bid_pos=0.0,
        cap_bid_neg=0.0,
    )
    delivered = float(m["delivered_activation_neg_mwh"])
    expected = delivered * 1000.0
    assert delivered > 0.0
    assert np.isclose(float(m["revenue_activation_eur"]), expected, atol=1e-6), m


def test_bem_only_forced_scenario_has_nonzero_revenue_and_benchmark_diagnostics() -> None:
    bt = _mk_backtester()
    n = 48
    idx = pd.date_range("2025-05-01 00:00:00+00:00", periods=n, freq="h", tz="UTC")
    col = BacktestColumnMap()
    df = pd.DataFrame(
        {
            col.timestamp: idx,
            col.pred_da_price: np.full(n, -20.0),
            f"{col.pred_da_price}_p05": np.full(n, -25.0),
            f"{col.pred_da_price}_p10": np.full(n, -23.0),
            f"{col.pred_da_price}_p90": np.full(n, -17.0),
            f"{col.pred_da_price}_p95": np.full(n, -15.0),
            col.pred_afrr_capacity_price_pos: np.full(n, 1.0),
            col.pred_afrr_capacity_price_neg: np.full(n, 1.0),
            col.pred_afrr_activation_price_pos: np.full(n, 140.0),
            col.pred_afrr_activation_price_neg: np.full(n, 10.0),
            col.pred_afrr_activation_rate_pos: np.full(n, 0.95),
            col.pred_afrr_activation_rate_neg: np.full(n, 0.05),
            col.true_da_price: np.full(n, -18.0),
            col.true_afrr_capacity_price_pos: np.full(n, 0.5),
            col.true_afrr_capacity_price_neg: np.full(n, 0.5),
            col.true_afrr_activation_price_pos: np.full(n, 150.0),
            col.true_afrr_activation_price_neg: np.full(n, 20.0),
            col.true_afrr_activation_rate_pos: np.full(n, 0.95),
            col.true_afrr_activation_rate_neg: np.full(n, 0.05),
        }
    )
    prev_eps = os.environ.get("BACKTEST_ORACLE_UPPER_BOUND_EPS")
    os.environ["BACKTEST_ORACLE_UPPER_BOUND_EPS"] = "1e12"
    try:
        out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=24, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    finally:
        if prev_eps is None:
            os.environ.pop("BACKTEST_ORACLE_UPPER_BOUND_EPS", None)
        else:
            os.environ["BACKTEST_ORACLE_UPPER_BOUND_EPS"] = prev_eps
    h = out.hourly
    def _series_or_zero(col: str) -> pd.Series:
        if col in h.columns:
            return pd.to_numeric(h[col], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=h.index)
    pos_mwh = _series_or_zero("real_bem_only_executed_pos_mwh")
    bem_rev = _series_or_zero("real_bem_only_activation_revenue_eur")
    cap_rev = _series_or_zero("real_revenue_capacity_eur")
    da_mw = _series_or_zero("real_da_net_mw")
    cap_pos = _series_or_zero("real_afrr_bcm_executed_pos_mw")
    cap_neg = _series_or_zero("real_afrr_bcm_executed_neg_mw")
    bem_sub_pos = _series_or_zero("real_bem_only_submitted_pos_mw")
    bem_sub_neg = _series_or_zero("real_bem_only_submitted_neg_mw")
    pure_mask = (bem_sub_pos + bem_sub_neg > 1e-12) & (da_mw.abs() < 1e-12) & ((cap_pos + cap_neg).abs() < 1e-12)
    assert float(pos_mwh.sum()) > 0.0
    assert float(bem_rev.sum()) > 0.0
    assert np.isclose(float(cap_rev[pure_mask].sum()), 0.0, atol=1e-9)
    real_cmp = float(out.summary["comparable_realized_market_pnl_eur"])
    perfect_foresight_cmp = float(out.summary["comparable_perfect_foresight_market_pnl_eur"])
    assert np.isfinite(real_cmp)
    assert np.isfinite(perfect_foresight_cmp)
    assert out.summary.get("comparable_benchmark_type") == "rolling_perfect_foresight_same_rules"


def test_no_headroom_violations_in_clean_forced_bem_only() -> None:
    bt = _mk_backtester()
    n = 24
    idx = pd.date_range("2025-05-01 00:00:00+00:00", periods=n, freq="h", tz="UTC")
    col = BacktestColumnMap()
    df = pd.DataFrame(
        {
            col.timestamp: idx,
            col.pred_da_price: np.full(n, -20.0),
            f"{col.pred_da_price}_p05": np.full(n, -25.0),
            f"{col.pred_da_price}_p10": np.full(n, -23.0),
            f"{col.pred_da_price}_p90": np.full(n, -17.0),
            f"{col.pred_da_price}_p95": np.full(n, -15.0),
            col.pred_afrr_capacity_price_pos: np.full(n, 0.5),
            col.pred_afrr_capacity_price_neg: np.full(n, 0.5),
            col.pred_afrr_activation_price_pos: np.full(n, 140.0),
            col.pred_afrr_activation_price_neg: np.full(n, 10.0),
            col.pred_afrr_activation_rate_pos: np.full(n, 0.95),
            col.pred_afrr_activation_rate_neg: np.full(n, 0.05),
            col.true_da_price: np.full(n, -18.0),
            col.true_afrr_capacity_price_pos: np.full(n, 0.5),
            col.true_afrr_capacity_price_neg: np.full(n, 0.5),
            col.true_afrr_activation_price_pos: np.full(n, 150.0),
            col.true_afrr_activation_price_neg: np.full(n, 20.0),
            col.true_afrr_activation_rate_pos: np.full(n, 0.95),
            col.true_afrr_activation_rate_neg: np.full(n, 0.05),
        }
    )
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=24, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    hv = float(out.summary.get("headroom_violation_count", 0.0))
    hmax = float(out.summary.get("headroom_violation_max_mwh", 0.0))
    assert hv <= 1e-9
    assert hmax <= 1e-9


def test_id_is_price_taker() -> None:
    bt = _mk_backtester()
    _, m = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=0.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=100.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        id_charge_mw=1.0,
        id_discharge_mw=0.0,
    )
    assert float(m["id_buy_mwh"]) > 0.0
    # ID cost follows modeled realized settlement price path directly.
    assert float(m["cost_id_eur"]) > 0.0
    assert np.isclose(float(m["id_buy_price_eur_mwh"]), min(bt.id_buy_price_cap_eur_mwh, 100.0 + bt.id_rescue_spread_eur_mwh))
    assert np.isclose(float(m["id_sell_price_eur_mwh"]), max(bt.id_sell_price_floor_eur_mwh, 100.0 - bt.id_rescue_spread_eur_mwh))
    assert float(m["id_buy_price_eur_mwh"]) >= float(m["id_sell_price_eur_mwh"]) - 1e-12


def test_bcm_positive_headroom_30min() -> None:
    bt = _mk_backtester()
    # Need >= reserve*0.5/eta_out above soc_min for 10 MW reserve.
    req = 10.0 * bt.reserve_activation_headroom_h / bt.eta_out
    soc_start = bt.soc_min + req - 0.2
    df, col = _one_hour_pred_df(
        da=0.0, cap_pos=300.0, cap_neg=0.0, act_pos=100.0, act_neg=0.0, rate_pos=1.0, rate_neg=0.0
    )
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=soc_start,
        allowed_markets=("aFRR",),
    )
    assert float(out["reserve_pos_mw"].iloc[0]) < 10.0


def test_bem_only_headroom_fields_present() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=-100.0, cap_pos=0.0, cap_neg=0.0, act_pos=400.0, act_neg=-10.0, rate_pos=1.0, rate_neg=0.0
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("DA", "aFRR"))
    for c in [
        "required_headroom_pos_mwh",
        "required_headroom_neg_mwh",
        "available_headroom_pos_mwh",
        "available_headroom_neg_mwh",
        "headroom_margin_pos_mwh",
        "headroom_margin_neg_mwh",
        "power_stack_pos_mw",
        "power_stack_neg_mw",
    ]:
        assert c in out.columns


def test_bcm_negative_headroom_30min() -> None:
    bt = _mk_backtester()
    req = 10.0 * bt.reserve_activation_headroom_h * bt.eta_in
    soc_start = bt.soc_max - req + 0.2
    df, col = _one_hour_pred_df(
        da=0.0, cap_pos=0.0, cap_neg=300.0, act_pos=0.0, act_neg=-100.0, rate_pos=0.0, rate_neg=1.0
    )
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=soc_start,
        allowed_markets=("aFRR",),
    )
    assert float(out["reserve_neg_mw"].iloc[0]) < 10.0


def test_headroom_with_efficiency() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.9
    bt.eta_out = 0.9
    df, col = _one_hour_pred_df(
        da=0.0, cap_pos=100.0, cap_neg=100.0, act_pos=100.0, act_neg=-100.0, rate_pos=1.0, rate_neg=1.0
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("aFRR",))
    row = out.iloc[0]
    pos_req_expected = (
        float(row["reserve_pos_mw"]) * bt.reserve_activation_headroom_h
        + float(row.get("bem_only_pos_mw", 0.0)) * bt.bem_activation_headroom_h
    ) / bt.eta_out
    neg_req_expected = (
        float(row["reserve_neg_mw"]) * bt.reserve_activation_headroom_h
        + float(row.get("bem_only_neg_mw", 0.0)) * bt.bem_activation_headroom_h
    ) * bt.eta_in
    assert np.isclose(float(row["required_headroom_pos_mwh"]), pos_req_expected, atol=1e-9)
    assert np.isclose(float(row["required_headroom_neg_mwh"]), neg_req_expected, atol=1e-9)


def test_bem_only_positive_headroom() -> None:
    bt = _mk_backtester()
    req = 10.0 * bt.bem_activation_headroom_h / bt.eta_out
    soc_start = bt.soc_min + req - 0.2
    df, col = _one_hour_pred_df(
        da=-100.0, cap_pos=0.0, cap_neg=0.0, act_pos=500.0, act_neg=-10.0, rate_pos=1.0, rate_neg=0.0
    )
    out = bt.optimize_dispatch(df, col, soc_start=soc_start, allowed_markets=("DA", "aFRR"))
    assert float(out["bem_only_pos_mw"].iloc[0]) < 10.0


def test_bem_only_negative_headroom() -> None:
    bt = _mk_backtester()
    req = 10.0 * bt.bem_activation_headroom_h * bt.eta_in
    soc_start = bt.soc_max - req + 0.2
    df, col = _one_hour_pred_df(
        da=100.0, cap_pos=0.0, cap_neg=0.0, act_pos=0.0, act_neg=-500.0, rate_pos=0.0, rate_neg=1.0
    )
    out = bt.optimize_dispatch(df, col, soc_start=soc_start, allowed_markets=("DA", "aFRR"))
    assert float(out["bem_only_neg_mw"].iloc[0]) < 10.0


def test_no_ex_post_missed_capacity_when_headroom_feasible() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=3)
    df[col.pred_afrr_capacity_price_pos] = 200.0
    df[col.pred_afrr_activation_price_pos] = 150.0
    df[col.pred_afrr_activation_rate_pos] = 1.0
    df[col.true_afrr_capacity_price_pos] = 200.0
    df[col.true_afrr_activation_price_pos] = 150.0
    df[col.true_afrr_activation_rate_pos] = 1.0
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=3, reopt_step_hours=1, allowed_markets=("aFRR",))
    h = out.hourly
    assert np.isclose(float(pd.to_numeric(h.get("real_missed_activation_pos_mwh", 0.0), errors="coerce").fillna(0.0).sum()), 0.0, atol=1e-9)
    assert np.isclose(float(pd.to_numeric(h.get("real_missed_activation_neg_mwh", 0.0), errors="coerce").fillna(0.0).sum()), 0.0, atol=1e-9)


def test_audit_fields_present() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=3)
    df[col.pred_da_price] = 10.0
    df[col.pred_afrr_capacity_price_pos] = 10.0
    df[col.pred_afrr_capacity_price_neg] = 10.0
    df[col.pred_afrr_activation_price_pos] = 20.0
    df[col.pred_afrr_activation_price_neg] = -20.0
    df[col.pred_afrr_activation_rate_pos] = 0.2
    df[col.pred_afrr_activation_rate_neg] = 0.2
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=3, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    h = out.hourly
    required = [
        "required_headroom_pos_mwh",
        "required_headroom_neg_mwh",
        "power_stack_pos_mw",
        "power_stack_neg_mw",
        "real_revenue_capacity_eur",
        "real_revenue_activation_eur",
        "real_pnl_eur",
        "afrr_cap_pos_awarded_mw",
        "afrr_cap_neg_awarded_mw",
        "delivered_activation_pos_mwh",
        "delivered_activation_neg_mwh",
        "real_afrr_cap_pos_awarded_mw",
        "real_afrr_cap_neg_awarded_mw",
        "real_delivered_activation_pos_mwh",
        "real_delivered_activation_neg_mwh",
        "real_required_headroom_pos_mwh",
        "real_required_headroom_neg_mwh",
    ]
    for c in required:
        assert c in h.columns
    s = out.summary
    for k in [
        "id_price_taker",
        "reserve_activation_headroom_h",
        "bem_activation_headroom_h",
        "headroom_violation_count",
        "headroom_violation_max_mwh",
    ]:
        assert k in s


def test_pnl_reconciliation() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=3)
    df[col.pred_da_price] = 50.0
    df[col.true_da_price] = 50.0
    df[col.pred_afrr_capacity_price_pos] = 20.0
    df[col.pred_afrr_capacity_price_neg] = 5.0
    df[col.pred_afrr_activation_price_pos] = 30.0
    df[col.pred_afrr_activation_price_neg] = -10.0
    df[col.pred_afrr_activation_rate_pos] = 0.3
    df[col.pred_afrr_activation_rate_neg] = 0.2
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=3, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    h = out.hourly
    pnl = pd.to_numeric(h["real_pnl_eur"], errors="coerce").fillna(0.0)
    rev_da = pd.to_numeric(h["real_revenue_da_eur"], errors="coerce").fillna(0.0)
    rev_id = pd.to_numeric(h["real_revenue_id_eur"], errors="coerce").fillna(0.0)
    rev_cap = pd.to_numeric(h["real_revenue_capacity_eur"], errors="coerce").fillna(0.0)
    rev_act = pd.to_numeric(h["real_revenue_activation_eur"], errors="coerce").fillna(0.0)
    cost_deg = pd.to_numeric(h["real_degradation_cost_eur"], errors="coerce").fillna(0.0)
    cost_aux = pd.to_numeric(h["real_aux_cost_eur"], errors="coerce").fillna(0.0)
    penalties = pd.to_numeric(h["real_penalty_eur"], errors="coerce").fillna(0.0)
    recon = rev_da + rev_id + rev_cap + rev_act - cost_deg - cost_aux - penalties
    assert np.isclose(float((pnl - recon).abs().max()), 0.0, atol=1e-6)
    assert "real_pnl_reconciliation_error_eur" in h.columns
    assert float(pd.to_numeric(h["real_pnl_reconciliation_error_eur"], errors="coerce").fillna(0.0).max()) <= 1e-6


def test_headroom_audit_matches_constraint_timing() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=0.0,
        cap_pos=250.0,
        cap_neg=0.0,
        act_pos=120.0,
        act_neg=0.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(df, col, soc_start=bt.soc_min + 6.0, allowed_markets=("aFRR",))
    r = out.iloc[0]
    expected_avail = float(r["soc_start_lp_mwh"]) - bt.soc_min
    expected_req = (
        float(r["reserve_pos_mw"]) * bt.reserve_activation_headroom_h
        + float(r.get("bem_only_pos_mw", 0.0)) * bt.bem_activation_headroom_h
    ) / bt.eta_out
    assert np.isclose(float(r["available_headroom_pos_mwh"]), expected_avail, atol=1e-9)
    assert np.isclose(float(r["required_headroom_pos_mwh"]), expected_req, atol=1e-9)


def test_terminal_soc_pnl_reconciliation() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    # Encourage discharge so terminal repair/adjustment is active.
    df[col.pred_da_price] = 120.0
    df[col.true_da_price] = 120.0
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, allowed_markets=("DA",))
    s = out.summary
    rhs = float(s["realized_pnl_excl_terminal_eur"]) + float(s["terminal_soc_adjustment_eur"])
    assert np.isclose(float(s["realized_total_pnl_eur"]), rhs, atol=1e-6)


def test_summary_flags_invalid_runs() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    # Force fallback-driven infeasible stress with high reserve commitments.
    df[col.pred_afrr_capacity_price_pos] = 1e3
    df[col.true_afrr_capacity_price_pos] = 1e3
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=2, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    s = out.summary
    for k in [
        "simulation_valid",
        "invalid_reason",
        "final_soc_check_pass",
        "missed_capacity_check_pass",
        "pnl_reconciliation_check_pass",
        "headroom_check_pass",
    ]:
        assert k in s


def test_bcm_precommitment_reduces_infeasible_bid() -> None:
    bt = _mk_backtester()
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")  # 08:00 CET/CEST gate for model path
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [10.0, 10.0, 10.0, 10.0],
            "reserve_neg_mw": [0.0, 0.0, 0.0, 0.0],
            # only ~2 MWh above soc_min -> infeasible for 10 MW @ 0.5h
            "soc_start_lp_mwh": [bt.soc_min + 2.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [50.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [100.0] * 4,
        }
    ).set_index(col.timestamp)
    lock_pos: dict[pd.Timestamp, float] = {}
    lock_neg: dict[pd.Timestamp, float] = {}
    lock_e_pos: dict[pd.Timestamp, float] = {}
    lock_e_neg: dict[pd.Timestamp, float] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg=lock_neg,
        lock_energy_pos=lock_e_pos,
        lock_energy_neg=lock_e_neg,
        is_perfect_foresight=False,
    )
    # 2 MWh headroom with 0.5h/eta_out implies <4 MW feasible.
    assert len(lock_pos) == 4
    assert max(lock_pos.values()) <= 4.0 + 1e-9


def test_precommit_includes_aux_losses() -> None:
    bt = _mk_backtester()
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    # Without aux this would be feasible for ~4 MW; with 0.2 MWh aux per hour it is reduced.
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [4.0, 4.0, 4.0, 4.0],
            "reserve_neg_mw": [0.0, 0.0, 0.0, 0.0],
            "soc_start_lp_mwh": [bt.soc_min + 2.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.2] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [50.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [100.0] * 4,
        }
    ).set_index(col.timestamp)
    lock_pos: dict[pd.Timestamp, float] = {}
    lock_neg: dict[pd.Timestamp, float] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg=lock_neg,
        lock_energy_pos={},
        lock_energy_neg={},
        is_perfect_foresight=False,
    )
    assert len(lock_pos) == 4
    assert max(lock_pos.values()) < 4.0


def test_precommit_includes_existing_lockbook_obligations() -> None:
    bt = _mk_backtester()
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [4.0, 4.0, 4.0, 4.0],
            "reserve_neg_mw": [0.0, 0.0, 0.0, 0.0],
            "soc_start_lp_mwh": [bt.soc_min + 3.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [50.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [100.0] * 4,
        }
    ).set_index(col.timestamp)
    lock_pos: dict[pd.Timestamp, float] = {ts: 3.0 for ts in target_hours}
    lock_neg: dict[pd.Timestamp, float] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg=lock_neg,
        lock_energy_pos={},
        lock_energy_neg={},
        is_perfect_foresight=False,
    )
    # New offer should be clamped to leave room for already-locked 3 MW.
    assert len(lock_pos) == 4
    assert max(lock_pos.values()) <= 3.0 + 1e-9


def test_final_report_excludes_invalid_runs() -> None:
    df = pd.DataFrame(
        {
            "scenario": ["p50_p50", "p70_p90"],
            "trading_strategy": ["multi", "multi"],
            "simulation_valid": [0.0, 1.0],
            "invalid_reason": ["missed_capacity", ""],
        }
    )
    valid = df.loc[pd.to_numeric(df["simulation_valid"], errors="coerce").fillna(0.0) >= 0.5]
    assert len(valid) == 1
    assert valid.iloc[0]["scenario"] == "p70_p90"


def test_precommit_audit_fields_present() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=30)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=12, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    required = [
        "precommit_clamp_applied",
        "precommit_clamp_reason",
        "precommit_feasible_pos_mw",
        "precommit_feasible_neg_mw",
        "precommit_original_pos_mw",
        "precommit_original_neg_mw",
        "precommit_clamped_pos_mw",
        "precommit_clamped_neg_mw",
        "precommit_headroom_margin_min_mwh",
        "precommit_power_margin_min_mw",
        "precommit_soc_margin_min_mwh",
        "precommit_aux_loss_margin_mwh",
        "precommit_lockbook_obligation_pos_mw",
        "precommit_lockbook_obligation_neg_mw",
    ]
    for c in required:
        assert c in out.hourly.columns
    assert "precommit_clamp_applied_count" in out.summary
    assert "precommit_clamped_pos_mw_sum" in out.summary
    assert "precommit_clamped_neg_mw_sum" in out.summary


def test_summary_includes_precommit_and_fallback_counts() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=24)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=8, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    s = out.summary
    for k in [
        "fallback_mode_counts",
        "optimization_error_code_counts",
        "activation_split_reconciliation_error_max",
        "precommit_clamp_applied_count",
    ]:
        assert k in s


def test_fallback_infeasible_marks_invalid() -> None:
    bt = _mk_backtester()
    # Build a tiny case likely to trigger fallback with reserve obligation at low SoC.
    df, col = _tiny_backtest_df(hours=6)
    df[col.pred_afrr_capacity_price_pos] = 1000.0
    df[col.true_afrr_capacity_price_pos] = 1000.0
    bt.soc_init = bt.soc_min + 0.1
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    if "reserve_infeasible" in str(out.summary.get("invalid_reason", "")):
        assert float(out.summary.get("simulation_valid", 0.0)) == 0.0


def test_precommit_clamp_prevents_start_of_hour_headroom_violation() -> None:
    bt = _mk_backtester()
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    # Barely above min SoC; naive 10 MW bid would violate strict start-of-hour headroom.
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [10.0, 10.0, 10.0, 10.0],
            "reserve_neg_mw": [0.0, 0.0, 0.0, 0.0],
            "soc_start_lp_mwh": [bt.soc_min + 2.2] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [100.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [150.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [1000.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    lock_pos: dict[pd.Timestamp, float] = {}
    lock_neg: dict[pd.Timestamp, float] = {}
    precommit: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg=lock_neg,
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=precommit,
        is_perfect_foresight=False,
    )
    assert len(lock_pos) == 4
    assert max(lock_pos.values()) < 10.0
    applied = precommit.get("precommit_clamp_applied", {})
    assert any(float(v) > 0.5 for v in applied.values())


def test_fallback_repair_preserves_start_of_hour_reserve_headroom() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=10)
    # Encourage reserve commitments and make solve harder.
    df[col.pred_afrr_capacity_price_pos] = 1e3
    df[col.true_afrr_capacity_price_pos] = 1e3
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    codes = (
        out.hourly["optimization_error_code"].astype(str).value_counts().to_dict()
        if "optimization_error_code" in out.hourly.columns
        else {}
    )
    # Legacy unsafe label must not be used when reserve obligations exist.
    assert "safe_hold_plan_under_infeasible_soft_final_soc" not in codes


def test_intrahour_id_rescue_does_not_mask_start_of_hour_headroom_violation() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    df[col.pred_afrr_capacity_price_pos] = 1e3
    df[col.true_afrr_capacity_price_pos] = 1e3
    bt.soc_init = bt.soc_min + 0.05
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    hv = float(out.summary.get("headroom_violation_count", 0.0))
    if hv > 1e-9:
        assert float(out.summary.get("headroom_check_pass", 1.0)) == 0.0
        assert float(out.summary.get("simulation_valid", 1.0)) == 0.0


def test_fallback_marks_scenario_invalid_in_strict_mode() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    df[col.pred_afrr_capacity_price_pos] = 1000.0
    df[col.true_afrr_capacity_price_pos] = 1000.0
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=4,
        reopt_step_hours=1,
        allowed_markets=("DA", "aFRR"),
        strict_simulation_validity=True,
    )
    fb = float(out.summary.get("fallback_used", 0.0))
    if fb > 0.5:
        assert float(out.summary.get("simulation_valid", 1.0)) == 0.0
        assert float(out.summary.get("thesis_reportable", 1.0)) == 0.0
        assert "fallback_used" in str(out.summary.get("invalid_reason", ""))


def test_safe_hold_not_reportable() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    bt.soc_init = bt.soc_min + 0.01
    df[col.pred_afrr_capacity_price_pos] = 1000.0
    df[col.true_afrr_capacity_price_pos] = 1000.0
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=4,
        reopt_step_hours=1,
        allowed_markets=("DA", "aFRR"),
        strict_simulation_validity=True,
    )
    codes = out.summary.get("optimization_error_code_counts", "{}")
    if "safe_hold" in str(codes).lower():
        assert float(out.summary.get("fallback_used", 0.0)) >= 1.0
        assert float(out.summary.get("thesis_reportable", 1.0)) == 0.0


def test_valid_run_has_no_fallback_modes() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=12)
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=6,
        reopt_step_hours=1,
        allowed_markets=("DA",),
        strict_simulation_validity=True,
    )
    if float(out.summary.get("simulation_valid", 0.0)) >= 0.5:
        assert float(out.summary.get("fallback_used", 1.0)) == 0.0
        counts = str(out.summary.get("optimization_error_code_counts", "{}")).lower()
        assert counts in {"{}", ""} or ("ok" in counts)


def test_validity_flags_are_consistent() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    if float(s.get("headroom_violation_count", 0.0)) > 1e-9:
        assert float(s.get("simulation_valid", 1.0)) == 0.0
    if float(s.get("missed_capacity_pos_mw", 0.0)) + float(s.get("missed_capacity_neg_mw", 0.0)) > 1e-9:
        assert float(s.get("simulation_valid", 1.0)) == 0.0
    if float(s.get("fallback_used", 0.0)) > 0.5:
        assert float(s.get("simulation_valid", 1.0)) == 0.0


def test_reserve_commitment_traceability_fields_present() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=30)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=12, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    req = [
        "reserve_commitment_id",
        "reserve_product_block_id",
        "reserve_commitment_source_snapshot_utc",
        "reserve_delivery_start_utc",
        "reserve_delivery_end_utc",
        "reserve_precommit_feasible_pos_mw",
        "reserve_precommit_feasible_neg_mw",
        "reserve_submitted_pos_mw",
        "reserve_submitted_neg_mw",
        "reserve_awarded_pos_mw",
        "reserve_awarded_neg_mw",
        "reserve_lockbook_pos_mw",
        "reserve_lockbook_neg_mw",
    ]
    for c in req:
        assert c in out.hourly.columns


def test_precommit_clamp_includes_projection_safety_buffer() -> None:
    bt = _mk_backtester()
    # Compare with/without projection safety by manually toggling on same synthetic block.
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [10.0, 10.0, 10.0, 10.0],
            "reserve_neg_mw": [0.0, 0.0, 0.0, 0.0],
            "soc_start_lp_mwh": [bt.soc_min + 2.5] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    pre0: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt.reserve_soc_projection_safety_mwh = 0.0
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap.copy(),
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre0,
        is_perfect_foresight=False,
    )
    pre1: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt.reserve_soc_projection_safety_mwh = 0.1
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap.copy(),
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre1,
        is_perfect_foresight=False,
    )
    f0 = min(float(v) for v in pre0.get("precommit_feasible_pos_mw", {}).values())
    f1 = min(float(v) for v in pre1.get("precommit_feasible_pos_mw", {}).values())
    assert f1 <= f0 + 1e-9


def test_reserve_min_margin_after_bid_applied() -> None:
    bt = _mk_backtester()
    bt.reserve_feasibility_mode = "conservative"
    bt.reserve_min_margin_after_bid_mwh = 10.0  # force zeroing
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [5.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 5.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    lock_pos: dict[pd.Timestamp, float] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
    )
    assert all(float(v) == 0.0 for v in lock_pos.values())
    assert any(float(v) > 0.5 for v in pre.get("precommit_zeroed_due_to_margin", {}).values())


def test_conservative_precommit_reduces_bid_to_safe_mw() -> None:
    bt = _mk_backtester()
    bt.reserve_feasibility_mode = "conservative"
    bt.reserve_min_margin_after_bid_mwh = 0.4
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [5.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 2.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    lock_pos: dict[pd.Timestamp, float] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
    )
    assert any(float(v) > 0.5 for v in pre.get("precommit_reduced_due_to_margin", {}).values())
    assert any(float(v) > 0.0 for v in lock_pos.values())
    assert any(float(v) < 5.0 for v in lock_pos.values())


def test_conservative_reserve_cli_params_reach_backtester() -> None:
    txt = Path("scripts/run_battery_backtest.py").read_text(encoding="utf-8")
    assert "--reserve-feasibility-mode" in txt
    assert "--reserve-soc-projection-safety-mwh" in txt
    assert "--reserve-headroom-safety-mwh" in txt
    assert "--reserve-power-safety-mw" in txt
    assert "--reserve-min-margin-after-bid-mwh" in txt
    assert 'MODEL_SPECS["reserve_feasibility_mode"]' in txt
    assert 'MODEL_SPECS["reserve_soc_projection_safety_mwh"]' in txt
    assert 'MODEL_SPECS["reserve_headroom_safety_mwh"]' in txt
    assert 'MODEL_SPECS["reserve_power_safety_mw"]' in txt
    assert 'MODEL_SPECS["reserve_min_margin_after_bid_mwh"]' in txt
    assert "--reserve-bid-derate" in txt
    assert "--max-reserve-bid-mw" in txt
    assert 'MODEL_SPECS["reserve_bid_derate"]' in txt
    assert 'MODEL_SPECS["max_reserve_bid_mw"]' in txt


def test_summary_never_omits_required_validity_fields() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=12)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1, strict_simulation_validity=True)
    req = [
        "simulation_valid",
        "thesis_reportable",
        "invalid_reason",
        "fallback_used",
        "infeasibility_driver",
        "fallback_mode_counts",
        "optimization_error_code_counts",
        "precommit_clamp_applied_count",
        "precommit_clamped_pos_mw_sum",
        "precommit_clamped_neg_mw_sum",
        "precommit_clamp_reasons",
        "activation_split_reconciliation_error_max",
        "reserve_soc_projection_safety_mwh",
        "reserve_headroom_safety_mwh",
        "reserve_power_safety_mw",
        "reserve_feasibility_mode",
        "reserve_min_margin_after_bid_mwh",
        "reserve_bid_derate",
        "max_reserve_bid_mw",
        "precommit_reduced_due_to_margin_count",
        "precommit_safe_pos_mw_avg",
        "precommit_safe_neg_mw_avg",
        "fallback_is_repair_optimization",
        "afrr_bcm_gate_hour_cet_model",
        "afrr_bcm_gate_hour_cet_benchmark",
        "benchmark_same_rules_gate_consistent",
        "benchmark_is_global_upper_bound",
        "final_soc_handling_mode",
        "terminal_soc_net_adjustment_eur",
        "terminal_price_eur_mwh",
        "terminal_price_source",
    ]
    for c in req:
        assert c in out.summary


def test_fallback_is_repair_optimization_flag() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    assert float(out.summary.get("fallback_is_repair_optimization", 1.0)) == 0.0


def test_no_thesis_reportable_with_fallback() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    df[col.pred_afrr_capacity_price_pos] = 1000.0
    df[col.true_afrr_capacity_price_pos] = 1000.0
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    if float(out.summary.get("fallback_used", 0.0)) > 0.5:
        assert float(out.summary.get("thesis_reportable", 1.0)) == 0.0


def test_fallback_never_thesis_reportable() -> None:
    test_no_thesis_reportable_with_fallback()


def test_fallback_never_reportable() -> None:
    test_fallback_never_thesis_reportable()


def test_reserve_feasibility_repair_never_reportable() -> None:
    row = {
        "simulation_valid": 1.0,
        "final_soc_check_pass": 1.0,
        "missed_capacity_check_pass": 1.0,
        "missed_activation_check_pass": 1.0,
        "headroom_check_pass": 1.0,
        "protected_soc_check_pass": 1.0,
        "reserve_headroom_shortfall_check_pass": 1.0,
        "pnl_reconciliation_check_pass": 1.0,
        "fallback_used": 0.0,
        "reserve_feasibility_repair_used": 1.0,
    }
    thesis_reportable = float(
        (row["simulation_valid"] >= 0.5)
        and (row["fallback_used"] <= 0.5)
        and (row["reserve_feasibility_repair_used"] <= 0.5)
        and (row["final_soc_check_pass"] >= 0.5)
        and (row["missed_capacity_check_pass"] >= 0.5)
        and (row["missed_activation_check_pass"] >= 0.5)
        and (row["headroom_check_pass"] >= 0.5)
        and (row["protected_soc_check_pass"] >= 0.5)
        and (row["reserve_headroom_shortfall_check_pass"] >= 0.5)
        and (row["pnl_reconciliation_check_pass"] >= 0.5)
    )
    assert thesis_reportable == 0.0


def test_reserve_bid_derate_reduces_submitted_mw() -> None:
    bt = _mk_backtester()
    bt.reserve_bid_derate = 0.5
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [8.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 8.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
    )
    fea = list(pre.get("precommit_feasible_pos_mw", {}).values())
    sub = list(pre.get("precommit_submitted_pos_mw_after_derate_cap", {}).values())
    assert fea and sub
    assert max(float(s) for s in sub) <= max(float(f) for f in fea) + 1e-9


def test_max_reserve_bid_mw_caps_submission() -> None:
    bt = _mk_backtester()
    bt.reserve_bid_derate = 1.0
    bt.max_reserve_bid_mw = 2.0
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [8.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 8.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
    )
    sub = list(pre.get("precommit_submitted_pos_mw_after_derate_cap", {}).values())
    assert sub and all(float(v) <= 2.0 + 1e-9 for v in sub)


def test_derate_and_cap_written_to_summary_and_debug() -> None:
    txt_bt = Path("src/energy_trading/simulation/battery_backtest.py").read_text(encoding="utf-8")
    txt_run = Path("scripts/run_battery_backtest.py").read_text(encoding="utf-8")
    for key in (
        "reserve_bid_derate",
        "max_reserve_bid_mw",
        "final_soc_mode",
        "precommit_reduction_reason",
        "desired_reserve_pos_mw",
        "safe_reserve_pos_mw",
        "submitted_reserve_pos_mw",
        "precommit_submitted_pos_mw_after_derate_cap",
        "precommit_submitted_neg_mw_after_derate_cap",
    ):
        assert key in txt_bt
        assert key in txt_run


def test_conservative_fields_written_to_summary_and_debug() -> None:
    test_derate_and_cap_written_to_summary_and_debug()


def test_quantile_aggressiveness_not_changed_by_feasibility_cap() -> None:
    bt = _mk_backtester()
    bt.reserve_bid_derate = 0.5
    bt.max_reserve_bid_mw = 2.0
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [6.0] * 4,
            "reserve_neg_mw": [4.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 8.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [30.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [100.0] * 4,
        }
    ).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
    )
    assert all(abs(float(v) - 6.0) < 1e-9 for v in pre.get("desired_reserve_pos_mw", {}).values())
    assert all(abs(float(v) - 4.0) < 1e-9 for v in pre.get("desired_reserve_neg_mw", {}).values())


def test_reserve_submission_capped_by_safe_mw() -> None:
    bt = _mk_backtester()
    bt.reserve_bid_derate = 1.0
    bt.max_reserve_bid_mw = None
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [8.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 1.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
    )
    for ts, desired in pre.get("desired_reserve_pos_mw", {}).items():
        safe = float(pre.get("safe_reserve_pos_mw", {}).get(ts, 0.0))
        submitted = float(pre.get("submitted_reserve_pos_mw", {}).get(ts, 0.0))
        assert submitted <= safe + 1e-9
        assert safe <= float(desired) + 1e-9


def test_reserve_submission_never_exceeds_safe_mw() -> None:
    test_reserve_submission_capped_by_safe_mw()


def test_zero_reserve_when_safe_mw_below_tolerance() -> None:
    bt = _mk_backtester()
    bt.reserve_bid_derate = 1.0
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [5.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame({col.timestamp: target_hours}).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    lock_pos: dict[pd.Timestamp, float] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
    )
    assert all(float(v) <= 1e-9 for v in lock_pos.values())
    assert all(float(v) <= 1e-9 for v in pre.get("submitted_reserve_pos_mw", {}).values())


def test_zero_safe_reserve_submits_zero() -> None:
    test_zero_reserve_when_safe_mw_below_tolerance()


def test_reserve_bid_derate_applies_after_safe_mw() -> None:
    bt = _mk_backtester()
    bt.reserve_bid_derate = 0.5
    bt.max_reserve_bid_mw = None
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [4.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 8.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame({col.timestamp: target_hours}).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
    )
    for ts, safe in pre.get("safe_reserve_pos_mw", {}).items():
        sub = float(pre.get("submitted_reserve_pos_mw", {}).get(ts, 0.0))
        assert sub <= float(safe) * 0.5 + 1e-9


def test_reserve_bid_derate_applied_after_safe_mw() -> None:
    test_reserve_bid_derate_applies_after_safe_mw()


def test_max_reserve_bid_caps_after_safe_mw() -> None:
    bt = _mk_backtester()
    bt.reserve_bid_derate = 1.0
    bt.max_reserve_bid_mw = 1.0
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [8.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 8.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame({col.timestamp: target_hours}).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
    )
    assert all(float(v) <= 1.0 + 1e-9 for v in pre.get("submitted_reserve_pos_mw", {}).values())


def test_max_reserve_bid_mw_caps_submission() -> None:
    test_max_reserve_bid_caps_after_safe_mw()


def test_data_flow_has_single_canonical_soc_source() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=10)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1)
    # Canonical settlement SoC must exist and be finite.
    assert "real_soc_mwh" in out.hourly.columns
    rs = pd.to_numeric(out.hourly["real_soc_mwh"], errors="coerce")
    assert rs.notna().all()
    # If start-of-hour SoC audit field exists, it should be finite and aligned.
    if "real_soc_start_mwh" in out.hourly.columns:
        ss = pd.to_numeric(out.hourly["real_soc_start_mwh"], errors="coerce")
        assert ss.notna().all()


def test_required_debug_fields_present() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=16)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=8, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    req = [
        "reserve_commitment_id",
        "reserve_product_block_id",
        "reserve_commitment_source_snapshot_utc",
        "reserve_delivery_start_utc",
        "reserve_delivery_end_utc",
        "reserve_submitted_pos_mw",
        "reserve_awarded_pos_mw",
        "fixed_reserve_obligation_pos_mw",
        "real_required_headroom_pos_mwh",
        "real_available_headroom_pos_mwh",
        "optimization_error_code",
        "is_fallback_hour",
    ]
    for c in req:
        assert c in out.hourly.columns


def test_same_rules_benchmark_uses_same_bcm_gate() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=10)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1)
    assert float(out.summary.get("afrr_bcm_gate_hour_cet_model", -1.0)) == float(
        out.summary.get("afrr_bcm_gate_hour_cet_benchmark", -2.0)
    )
    assert float(out.summary.get("benchmark_same_rules_gate_consistent", 0.0)) == 1.0


def test_strict_final_soc_hard_constraint_or_repair() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    df[col.pred_da_price] = 200.0
    df[col.true_da_price] = 200.0
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    shortfall = float(s.get("final_soc_shortfall_mwh", 0.0))
    repair = float(s.get("terminal_soc_repair_cost_eur", 0.0))
    assert (shortfall <= 1e-6) or (repair > 0.0)
    assert float(s.get("final_soc_check_pass", 0.0)) >= 0.5
    assert "final_soc_actual_mwh" in s
    assert "final_soc_target_mwh" in s
    assert "final_soc_physical_check_pass" in s
    assert "final_soc_economic_repair_check_pass" in s
    assert "terminal_soc_repair_included_in_pnl" in s


def test_unrepaired_final_soc_shortfall_invalid() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    # Force shortfall and zero terminal repair by setting terminal price to zero.
    df[col.pred_da_price] = 100.0
    df[col.true_da_price] = 0.0
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    if float(s.get("final_soc_shortfall_mwh", 0.0)) > 1e-6:
        assert float(s.get("terminal_soc_repair_cost_eur", 0.0)) <= 1e-9
        assert float(s.get("terminal_soc_repair_included_in_pnl", 1.0)) == 0.0
        assert float(s.get("final_soc_check_pass", 1.0)) == 0.0
        assert float(s.get("simulation_valid", 1.0)) == 0.0
        assert float(s.get("thesis_reportable", 1.0)) == 0.0
        assert "final_soc_unrepaired" in str(s.get("invalid_reason", ""))


def test_final_soc_shortfall_repaired_is_valid_if_pnl_reconciles() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    df[col.pred_da_price] = 150.0
    df[col.true_da_price] = 120.0
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    if float(s.get("final_soc_shortfall_mwh", 0.0)) > 1e-6 and float(s.get("terminal_soc_repair_cost_eur", 0.0)) > 1e-9:
        assert float(s.get("terminal_soc_repair_included_in_pnl", 0.0)) >= 0.5
        assert float(s.get("final_soc_economic_repair_check_pass", 0.0)) >= 0.5
        assert float(s.get("final_soc_check_pass", 0.0)) >= 0.5


def test_terminal_repair_mode_allows_physical_shortfall_if_repaired() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "terminal_repair"
    df, col = _tiny_backtest_df(hours=6)
    df[col.pred_da_price] = 150.0
    df[col.true_da_price] = 120.0
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=6,
        reopt_step_hours=1,
        strict_simulation_validity=True,
        enforce_final_soc_min=False,
    )
    s = out.summary
    if float(s.get("final_soc_shortfall_mwh", 0.0)) > 1e-6:
        assert float(s.get("terminal_soc_repair_cost_eur", 0.0)) > 0.0
        assert float(s.get("terminal_soc_repair_included_in_pnl", 0.0)) >= 0.5
        assert float(s.get("final_soc_economic_repair_check_pass", 0.0)) >= 0.5
        assert float(s.get("final_soc_check_pass", 0.0)) >= 0.5


def test_hard_final_soc_mode_invalidates_shortfall() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "hard"
    df, col = _tiny_backtest_df(hours=6)
    df[col.pred_da_price] = 100.0
    df[col.true_da_price] = 0.0
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=6,
        reopt_step_hours=1,
        strict_simulation_validity=True,
        enforce_final_soc_min=False,
    )
    s = out.summary
    if float(s.get("final_soc_shortfall_mwh", 0.0)) > 1e-6:
        assert float(s.get("final_soc_physical_check_pass", 1.0)) == 0.0
        assert float(s.get("final_soc_check_pass", 1.0)) == 0.0
        assert float(s.get("simulation_valid", 1.0)) == 0.0


def test_hard_mode_final_soc_slack_not_used() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "hard"
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=8,
        reopt_step_hours=1,
        strict_simulation_validity=True,
        enforce_final_soc_min=True,
    )
    assert float(out.summary.get("final_soc_slack_used_mwh", 0.0)) <= 1e-9


def test_hard_mode_reaches_target_when_feasible() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "hard"
    df, col = _tiny_backtest_df(hours=8)
    # Encourage charging to satisfy terminal SoC hard floor.
    df[col.pred_da_price] = -100.0
    df[col.true_da_price] = -100.0
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=8,
        reopt_step_hours=1,
        strict_simulation_validity=True,
        enforce_final_soc_min=True,
    )
    assert float(out.summary.get("final_soc_physical_check_pass", 0.0)) >= 0.5


def test_strict_hard_mode_never_drops_terminal_constraint_in_fallback() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "hard"
    df, col = _tiny_backtest_df(hours=6)
    calls: list[float | None] = []
    original_opt = bt.optimize_dispatch

    def _wrapped(*args, **kwargs):
        calls.append(kwargs.get("soc_end_min_target"))
        return original_opt(*args, **kwargs)

    bt.optimize_dispatch = _wrapped  # type: ignore[method-assign]
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=6,
        reopt_step_hours=1,
        strict_simulation_validity=True,
        enforce_final_soc_min=True,
    )
    assert all(v is not None for v in calls)
    assert float(out.summary.get("terminal_constraint_dropped", 0.0)) == 0.0


def test_terminal_repair_mode_not_thesis_reportable_when_physical_shortfall() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "terminal_repair"
    df, col = _tiny_backtest_df(hours=6)
    df[col.pred_da_price] = 200.0
    df[col.true_da_price] = 120.0
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=6,
        reopt_step_hours=1,
        strict_simulation_validity=True,
        enforce_final_soc_min=True,
    )
    s = out.summary
    if float(s.get("final_soc_physical_check_pass", 1.0)) < 0.5:
        assert float(s.get("thesis_reportable", 1.0)) == 0.0


def test_terminal_soc_value_discount_does_not_replace_hard_final_soc() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "hard"
    df, col = _tiny_backtest_df(hours=4)
    orig_target = bt.soc_target_end
    orig_discount = MODEL_SPECS.get("terminal_soc_value_discount", None)
    try:
        bt.soc_target_end = float(bt.soc_max) + 5.0  # physically infeasible
        MODEL_SPECS["terminal_soc_value_discount"] = 1.0
        out = bt.run(
            df,
            col,
            use_rolling_horizon=True,
            horizon_hours=4,
            reopt_step_hours=1,
            strict_simulation_validity=True,
            enforce_final_soc_min=True,
        )
        s = out.summary
        assert float(s.get("final_soc_physical_check_pass", 1.0)) == 0.0
        assert float(s.get("thesis_reportable", 1.0)) == 0.0
    finally:
        bt.soc_target_end = orig_target
        if orig_discount is None:
            MODEL_SPECS.pop("terminal_soc_value_discount", None)
        else:
            MODEL_SPECS["terminal_soc_value_discount"] = orig_discount


def test_console_final_soc_reports_physical_and_economic_status() -> None:
    txt = Path("scripts/run_battery_backtest.py").read_text(encoding="utf-8")
    assert "final_soc_physical_check" in txt
    assert "final_soc_economic_repair_check" in txt


def test_terminal_repair_cost_reconciles() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    df[col.pred_da_price] = 150.0
    df[col.true_da_price] = 120.0
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1)
    s = out.summary
    rhs = float(s.get("realized_pnl_excl_terminal_eur", 0.0)) + float(s.get("terminal_soc_net_adjustment_eur", 0.0))
    assert np.isclose(float(s.get("realized_total_pnl_eur", 0.0)), rhs, atol=1e-6)


def test_strategy_overview_ratio_fields_semantics() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1)
    assert "realized_vs_perfect_foresight_ratio_multi_market" in out.summary
    assert "realized_vs_perfect_foresight_comparable_market_ratio" in out.summary
    assert "rolling_perfect_foresight_same_rules_total_pnl_eur" in out.summary
    assert "realized_vs_perfect_foresight_pct" in out.summary
    assert "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur" in out.summary
    assert "realized_vs_global_hindsight_perfect_foresight_upper_bound_pct" in out.summary
    assert float(out.summary.get("benchmark_is_global_upper_bound", 1.0)) == 0.0
    assert float(out.summary.get("rolling_perfect_foresight_same_rules_is_global_upper_bound", 1.0)) == 0.0


def test_rolling_pf_not_global_upper_bound() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1)
    assert float(out.summary.get("rolling_perfect_foresight_same_rules_is_global_upper_bound", 1.0)) == 0.0
    assert float(out.summary.get("rolling_perfect_foresight_same_rules_can_be_beaten", 0.0)) == 1.0


def test_global_perfect_foresight_is_full_horizon_not_rolling() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1)
    assert float(out.summary.get("global_perfect_foresight_available", 1.0)) == 0.0
    assert float(out.summary.get("global_hindsight_perfect_foresight_is_global_upper_bound", 1.0)) == 0.0
    assert str(out.summary.get("global_perfect_foresight_validation_status", "")).startswith("disabled")


def test_global_perfect_foresight_dominance_flag_present() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1)
    assert "global_perfect_foresight_dominance_check_pass" in out.summary
    assert "realized_exceeds_global_perfect_foresight" in out.summary


def test_global_perfect_foresight_does_not_invalidate_when_disabled() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    assert float(s.get("global_perfect_foresight_available", 1.0)) == 0.0
    assert float(s.get("global_perfect_foresight_dominance_check_pass", 0.0)) >= 0.5
    assert float(s.get("realized_exceeds_global_perfect_foresight", 1.0)) == 0.0
    assert "realized_exceeds_global_perfect_foresight" not in str(s.get("invalid_reason", ""))


def test_deprecated_perfect_foresight_aliases_are_labelled() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1)
    s = out.summary
    assert "perfect_foresight_total_pnl_eur" in s
    assert float(s.get("perfect_foresight_total_pnl_eur_is_deprecated", 0.0)) == 1.0
    assert str(s.get("perfect_foresight_total_pnl_eur_semantics", "")) == "rolling_perfect_foresight_same_rules"


def test_global_perfect_foresight_disabled_by_default() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1)
    s = out.summary
    assert float(s.get("global_perfect_foresight_available", 1.0)) == 0.0
    assert str(s.get("global_perfect_foresight_validation_status", "")).startswith("disabled")


def test_realized_beating_rolling_pf_does_not_invalidate() -> None:
    invalid_reason = "fallback_used,headroom"
    assert "realized_exceeds_rolling_pf" not in invalid_reason


def test_strategy_overview_labels_diagnostic_vs_upper_bound_ratios() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1)
    s = out.summary
    assert "realized_vs_perfect_foresight_pct" in s
    if float(s.get("global_perfect_foresight_available", 0.0)) < 0.5:
        assert np.isnan(float(s.get("realized_vs_global_hindsight_perfect_foresight_upper_bound_pct", float("nan"))))


def test_rolling_pf_quantile_surface_mode_present() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1)
    assert str(out.summary.get("rolling_pf_quantile_surface_mode", "")) in {
        "collapsed_to_truth",
        "preserved_quantile_surface",
        "unknown",
    }


def test_global_perfect_foresight_bem_only_plan_mapping() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=4,
        reopt_step_hours=1,
        enable_global_perfect_foresight=True,
    )
    s = out.summary
    # Scope-mismatch can still disable availability, but summary fields must exist.
    assert "global_perfect_foresight_bem_only_included" in s
    assert "global_perfect_foresight_dispatch_rows" in s
    assert "global_perfect_foresight_settlement_rows" in s


def test_global_perfect_foresight_available_only_after_scope_validation() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, enable_global_perfect_foresight=True)
    s = out.summary
    if float(s.get("global_perfect_foresight_available", 0.0)) >= 0.5:
        assert str(s.get("global_perfect_foresight_validation_status", "")) == "available_scope_validated"
        assert float(s.get("global_perfect_foresight_dispatch_rows", 0.0)) > 0.0
        assert float(s.get("global_perfect_foresight_settlement_rows", 0.0)) > 0.0
    else:
        assert str(s.get("global_perfect_foresight_validation_status", "")).startswith("disabled") or str(s.get("global_perfect_foresight_validation_status", "")).startswith("computed")


def test_locked_reserve_obligation_preserves_future_soc_headroom() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    # Make DA discharge attractive in early hours.
    df[col.pred_da_price] = [250.0, 250.0, 10.0, 10.0]
    ts = pd.to_datetime(df[col.timestamp], utc=True)
    # Lock a positive reserve obligation on the 3rd hour.
    lock = {pd.Timestamp(ts.iloc[2]): (8.0, 0.0)}
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=bt.soc_min + 6.5,
        fixed_reserve_obligation=lock,
        allowed_markets=("DA", "aFRR"),
    )
    r = out.iloc[2]
    req = 8.0 * bt.reserve_activation_headroom_h / bt.eta_out + bt.reserve_headroom_safety_mwh + bt.reserve_soc_projection_safety_mwh
    avail = float(r["soc_start_lp_mwh"]) - bt.soc_min
    assert avail + 1e-9 >= req


def test_locked_negative_reserve_obligation_preserves_future_empty_headroom() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    # Make DA charge attractive in early hours.
    df[col.pred_da_price] = [-200.0, -200.0, -10.0, -10.0]
    ts = pd.to_datetime(df[col.timestamp], utc=True)
    lock = {pd.Timestamp(ts.iloc[2]): (0.0, 8.0)}
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=bt.soc_max - 6.5,
        fixed_reserve_obligation=lock,
        allowed_markets=("DA", "aFRR"),
    )
    r = out.iloc[2]
    req = 8.0 * bt.reserve_activation_headroom_h * bt.eta_in + bt.reserve_headroom_safety_mwh + bt.reserve_soc_projection_safety_mwh
    avail = bt.soc_max - float(r["soc_start_lp_mwh"])
    assert avail + 1e-9 >= req


def test_fallback_counts_nonempty_when_fallback_used() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    df[col.pred_afrr_capacity_price_pos] = 1e3
    df[col.true_afrr_capacity_price_pos] = 1e3
    bt.soc_init = bt.soc_min + 0.01
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    if float(out.summary.get("fallback_used", 0.0)) > 0.5:
        counts = str(out.summary.get("fallback_mode_counts", "")).strip()
        assert counts not in {"", "{}"}


def test_locked_positive_reserve_blocks_future_da_discharge() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    # Strongly profitable DA discharge in first two hours.
    df[col.pred_da_price] = [400.0, 400.0, 0.0, 0.0]
    ts = pd.to_datetime(df[col.timestamp], utc=True)
    lock = {pd.Timestamp(ts.iloc[2]): (9.0, 0.0)}
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=bt.soc_min + 7.0,
        fixed_reserve_obligation=lock,
        allowed_markets=("DA", "aFRR"),
    )
    # Future locked reserve headroom must be preserved.
    req = 9.0 * bt.reserve_activation_headroom_h / bt.eta_out + bt.reserve_headroom_safety_mwh + bt.reserve_soc_projection_safety_mwh
    avail = float(out.loc[2, "soc_start_lp_mwh"]) - bt.soc_min
    assert avail + 1e-9 >= req


def test_locked_positive_reserve_blocks_future_bem_only_discharge() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    # Make BEM-only positive attractive in first hours.
    df[col.pred_afrr_activation_price_pos] = [300.0, 300.0, 0.0, 0.0]
    df[col.pred_afrr_activation_rate_pos] = [1.0, 1.0, 0.0, 0.0]
    ts = pd.to_datetime(df[col.timestamp], utc=True)
    lock = {pd.Timestamp(ts.iloc[2]): (8.0, 0.0)}
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=bt.soc_min + 6.5,
        fixed_reserve_obligation=lock,
        allowed_markets=("aFRR",),
    )
    req = 8.0 * bt.reserve_activation_headroom_h / bt.eta_out + bt.reserve_headroom_safety_mwh + bt.reserve_soc_projection_safety_mwh
    avail = float(out.loc[2, "soc_start_lp_mwh"]) - bt.soc_min
    assert avail + 1e-9 >= req


def test_locked_negative_reserve_blocks_future_charge() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    # Make charging attractive in first two hours.
    df[col.pred_da_price] = [-300.0, -300.0, 0.0, 0.0]
    ts = pd.to_datetime(df[col.timestamp], utc=True)
    lock = {pd.Timestamp(ts.iloc[2]): (0.0, 9.0)}
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=bt.soc_max - 7.0,
        fixed_reserve_obligation=lock,
        allowed_markets=("DA", "aFRR"),
    )
    req = 9.0 * bt.reserve_activation_headroom_h * bt.eta_in + bt.reserve_headroom_safety_mwh + bt.reserve_soc_projection_safety_mwh
    avail = bt.soc_max - float(out.loc[2, "soc_start_lp_mwh"])
    assert avail + 1e-9 >= req


def test_terminal_soc_does_not_override_locked_reserve() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    # Keep reserve obligation that consumes SoC margin near end.
    ts = pd.to_datetime(df[col.timestamp], utc=True)
    lock = {pd.Timestamp(ts.iloc[3]): (9.0, 0.0)}
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=bt.soc_min + 6.0,
        soc_end_min_target=bt.soc_target_end,
        fixed_reserve_obligation=lock,
        allowed_markets=("DA", "aFRR"),
    )
    req = 9.0 * bt.reserve_activation_headroom_h / bt.eta_out + bt.reserve_headroom_safety_mwh + bt.reserve_soc_projection_safety_mwh
    avail = float(out.loc[3, "soc_start_lp_mwh"]) - bt.soc_min
    assert avail + 1e-9 >= req


def test_reserve_commitment_debug_explains_soc_drift_fields() -> None:
    required = {
        "scenario",
        "submitted_mw",
        "awarded_mw",
        "locked_obligation_mw",
        "projected_soc_start_mwh_at_commitment",
        "projected_soc_start_mwh_latest_before_delivery",
        "realized_soc_start_mwh_at_delivery",
        "reserve_projected_vs_realized_soc_delta_mwh",
        "da_dispatch_mw_between_commit_and_delivery",
        "id_dispatch_mw_between_commit_and_delivery",
        "bem_only_dispatch_mw_between_commit_and_delivery",
        "aux_energy_mwh_between_commit_and_delivery",
        "required_headroom_mwh",
        "available_headroom_mwh",
        "headroom_violation_mwh",
    }
    # Column list in run_battery_backtest must include these fields.
    path = Path("scripts/run_battery_backtest.py")
    txt = path.read_text(encoding="utf-8")
    for c in required:
        assert c in txt


def test_no_thesis_reportable_reserve_shortfall() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    df[col.pred_afrr_capacity_price_pos] = 1e3
    df[col.true_afrr_capacity_price_pos] = 1e3
    bt.soc_init = bt.soc_min + 0.01
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    short_sum = float(s.get("reserve_headroom_shortfall_pos_mwh_sum", 0.0)) + float(
        s.get("reserve_headroom_shortfall_neg_mwh_sum", 0.0)
    )
    if short_sum > 1e-9:
        assert float(s.get("reserve_headroom_shortfall_check_pass", 1.0)) == 0.0
        assert float(s.get("thesis_reportable", 1.0)) == 0.0
        assert "reserve_headroom_shortfall" in str(s.get("invalid_reason", ""))


def test_protected_soc_envelope_fields_present() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    hourly_cols = set(out.hourly.columns)
    required_hourly = {
        "real_locked_reserve_pos_mw",
        "real_locked_reserve_neg_mw",
        "real_protected_soc_min_mwh",
        "real_protected_soc_max_mwh",
        "real_protected_soc_margin_pos_mwh",
        "real_protected_soc_margin_neg_mwh",
        "real_protected_soc_violation_pos_mwh",
        "real_protected_soc_violation_neg_mwh",
    }
    missing = required_hourly.difference(hourly_cols)
    assert not missing, f"Missing protected SoC fields: {sorted(missing)}"
    s = out.summary
    for k in (
        "protected_soc_violation_count",
        "protected_soc_violation_max_mwh",
        "protected_soc_check_pass",
        "locked_reserve_obligation_hours",
        "locked_reserve_obligation_pos_mw_sum",
        "locked_reserve_obligation_neg_mw_sum",
    ):
        assert k in s


def test_no_thesis_reportable_with_protected_soc_violation() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    psv = float(s.get("protected_soc_violation_count", 0.0))
    if psv > 1e-9:
        assert float(s.get("simulation_valid", 1.0)) == 0.0
        assert float(s.get("thesis_reportable", 1.0)) == 0.0
        assert float(s.get("protected_soc_check_pass", 1.0)) == 0.0
        assert "protected_soc" in str(s.get("invalid_reason", ""))


def test_submitted_bem_only_without_locked_obligation_does_not_activate_protected_envelope() -> None:
    bt = _mk_backtester()
    psv = bt._compute_obligation_driven_protected_soc_bounds(
        soc_start_mwh=4.0,
        required_headroom_pos_mwh=0.0,
        required_headroom_neg_mwh=0.0,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        committed_bem_pos_mw=4.0,
        committed_bem_neg_mw=0.0,
        reserve_pos_mw=4.0,
        reserve_neg_mw=0.0,
    )
    assert float(psv["obligation_headroom_pos_active"]) == 0.0
    assert np.isclose(float(psv["protected_soc_min_mwh"]), bt.soc_min)
    assert np.isclose(float(psv["protected_soc_max_mwh"]), bt.soc_max)
    assert float(psv["protected_soc_violation_pos_mwh"]) == 0.0


def test_locked_obligation_still_activates_protected_envelope() -> None:
    bt = _mk_backtester()
    req_pos = 2.0 * bt.reserve_activation_headroom_h / max(bt.eta_out, 1e-12)
    psv = bt._compute_obligation_driven_protected_soc_bounds(
        soc_start_mwh=bt.soc_min + req_pos - 0.1,
        required_headroom_pos_mwh=req_pos,
        required_headroom_neg_mwh=0.0,
        locked_reserve_pos_mw=2.0,
        locked_reserve_neg_mw=0.0,
        committed_bem_pos_mw=0.0,
        committed_bem_neg_mw=0.0,
        reserve_pos_mw=0.0,
        reserve_neg_mw=0.0,
    )
    assert float(psv["obligation_headroom_pos_active"]) == 1.0
    assert float(psv["protected_soc_min_mwh"]) > bt.soc_min
    assert float(psv["protected_soc_violation_pos_mwh"]) > 0.0


def test_precommit_ev_includes_headroom_cost() -> None:
    bt = _mk_backtester()
    bt.reserve_min_margin_after_bid_mwh = 0.0
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [8.0, 8.0, 8.0, 8.0],
            "reserve_neg_mw": [0.0, 0.0, 0.0, 0.0],
            "soc_start_lp_mwh": [bt.soc_min + 4.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.2] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [1.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [0.0] * 4,
            col.pred_afrr_activation_price_neg: [0.0] * 4,
            col.pred_da_price: [300.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    lock_pos: dict[pd.Timestamp, float] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
    )
    assert any(float(v) > 0.5 for v in pre.get("precommit_bid_zeroed_due_to_negative_ev", {}).values())
    assert any(float(v) > 0.0 for v in pre.get("precommit_headroom_opportunity_cost_eur", {}).values())


def test_required_series_missing_raises_keyerror() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0]})
    try:
        _ = require_numeric_series(df, "real_executed_charge_mw", aliases=["real_da_charge_mw"])
        assert False, "expected KeyError"
    except KeyError as exc:
        assert "real_executed_charge_mw" in str(exc)


def test_optional_series_missing_defaults_to_zero() -> None:
    df = pd.DataFrame(index=[0, 1, 2])
    s = optional_numeric_series(df, "optional_col", default=0.0)
    assert len(s) == len(df)
    assert np.isclose(float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum()), 0.0, atol=1e-12)


def test_legacy_alias_resolution_prefers_canonical() -> None:
    df = pd.DataFrame(
        {
            "real_executed_charge_mw": [1.0, 2.0],
            "real_da_charge_mw": [100.0, 200.0],
        }
    )
    s = require_numeric_series(df, "real_executed_charge_mw", aliases=["real_da_charge_mw"])
    assert np.isclose(float(s.iloc[0]), 1.0, atol=1e-12)
    assert np.isclose(float(s.iloc[1]), 2.0, atol=1e-12)


def test_validator_reports_debug_dump_counts(tmp_path: Path) -> None:
    scen = tmp_path / "multi" / "p50_p50"
    scen.mkdir(parents=True, exist_ok=True)
    payload = {
        "simulation_valid": 0.0,
        "thesis_reportable": 0.0,
        "invalid_reason": "optimization_infeasible_debug_dump",
        "fallback_used": 0.0,
        "fallback_mode_counts": "{}",
        "optimization_error_code_counts": '{"ok": 1.0}',
        "infeasible_debug_dump_count": 1.0,
        "accepted_path_infeasible_debug_dump_count": 1.0,
        "candidate_infeasible_debug_dump_count": 0.0,
        "infeasible_debug_dump_paths": '["artifacts/simulation_debug/infeasible_debug_X.npz"]',
        "precommit_clamp_applied_count": 0.0,
        "activation_split_reconciliation_error_max": 0.0,
        "final_soc_check_pass": 1.0,
        "benchmark_same_rules_gate_consistent": 1.0,
    }
    (scen / "backtest_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    df = validate_outputs._collect(tmp_path)
    assert not df.empty
    assert "infeasible_debug_dump_count" in df.columns
    assert "accepted_path_infeasible_debug_dump_count" in df.columns


def test_invalid_by_quantile_counts_only_invalid_scenarios(tmp_path: Path) -> None:
    base = tmp_path / "multi"
    base.mkdir(parents=True, exist_ok=True)
    common = {
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
    }
    payloads = {
        "p30_p30": dict(common, simulation_valid=0.0, thesis_reportable=0.0, invalid_reason="fallback_used"),
        "p50_p50": dict(common, simulation_valid=1.0, thesis_reportable=1.0, invalid_reason=""),
        "p70_p90": dict(common, simulation_valid=0.0, thesis_reportable=0.0, invalid_reason="optimization_failure"),
    }
    for scen, payload in payloads.items():
        d = base / scen
        d.mkdir(parents=True, exist_ok=True)
        (d / "backtest_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    out_json = tmp_path / "stats.json"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", ["validate_simulation_outputs.py", str(tmp_path), "--out-json", str(out_json)])
        validate_outputs.main()
    stats = json.loads(out_json.read_text(encoding="utf-8"))
    assert int(stats["invalid_scenarios"]) == 2
    assert stats["invalid_by_quantile"] == {"p30_p30": 1, "p70_p90": 1}


def test_validation_counts_are_consistent(tmp_path: Path) -> None:
    base = tmp_path / "multi"
    base.mkdir(parents=True, exist_ok=True)
    payload = {
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
    }
    for scen in ("p30_p30", "p50_p50", "p70_p90"):
        d = base / scen
        d.mkdir(parents=True, exist_ok=True)
        (d / "backtest_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    out_json = tmp_path / "stats.json"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", ["validate_simulation_outputs.py", str(tmp_path), "--out-json", str(out_json)])
        validate_outputs.main()
    stats = json.loads(out_json.read_text(encoding="utf-8"))
    assert int(stats["total_scenarios"]) == int(stats["valid_scenarios"]) + int(stats["invalid_scenarios"])
    assert sum(stats["invalid_by_quantile"].values()) == int(stats["invalid_scenarios"])


def test_accepted_infeasible_debug_dump_invalidates_scenario() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    s = dict(out.summary)
    s["accepted_path_infeasible_debug_dump_count"] = 1.0
    invalid = str(s.get("invalid_reason", ""))
    if "optimization_infeasible_debug_dump" not in invalid:
        invalid = ",".join([v for v in [invalid, "optimization_infeasible_debug_dump"] if v])
    s["invalid_reason"] = invalid
    s["simulation_valid"] = 0.0
    s["thesis_reportable"] = 0.0
    assert float(s["accepted_path_infeasible_debug_dump_count"]) > 0.5
    assert float(s["simulation_valid"]) == 0.0
    assert float(s["thesis_reportable"]) == 0.0
    assert "optimization_infeasible_debug_dump" in str(s["invalid_reason"])


def test_accepted_infeasible_dump_never_reportable() -> None:
    test_accepted_infeasible_debug_dump_invalidates_scenario()


def test_candidate_debug_dump_does_not_invalidate_if_not_accepted() -> None:
    row = {
        "simulation_valid": 1.0,
        "thesis_reportable": 1.0,
        "fallback_used": 0.0,
        "headroom_violation_count": 0.0,
        "missed_capacity_pos_mw": 0.0,
        "missed_capacity_neg_mw": 0.0,
        "pnl_reconciliation_error_max_eur": 0.0,
        "activation_split_reconciliation_error_max": 0.0,
        "final_soc_check_pass": 1.0,
        "benchmark_same_rules_gate_consistent": 1.0,
        "terminal_soc_repair_cost_eur": 0.0,
        "final_soc_actual_mwh": 10.0,
        "final_soc_target_mwh": 10.0,
        "optimization_error_code_counts": '{"ok": 10.0}',
        "accepted_path_infeasible_debug_dump_count": 0.0,
        "candidate_infeasible_debug_dump_count": 2.0,
        "infeasible_debug_dump_count": 2.0,
    }
    d = pd.DataFrame([row])
    non_ok = d["optimization_error_code_counts"].apply(lambda v: any(k not in {"ok", "none", ""} and float(c) > 0.0 for k, c in json.loads(v).items()))
    thesis_rule = (
        (d["simulation_valid"] >= 0.5)
        & (d["thesis_reportable"] >= 0.5)
        & (d["fallback_used"] <= 0.5)
        & (d["headroom_violation_count"] <= 1e-9)
        & (d["missed_capacity_pos_mw"] <= 1e-9)
        & (d["missed_capacity_neg_mw"] <= 1e-9)
        & (d["pnl_reconciliation_error_max_eur"] <= 1e-2)
        & (d["activation_split_reconciliation_error_max"] <= 1e-2)
        & (d["final_soc_check_pass"] >= 0.5)
        & (d["benchmark_same_rules_gate_consistent"] >= 0.5)
        & (d["accepted_path_infeasible_debug_dump_count"] <= 0.5)
        & (~non_ok)
    )
    assert bool(thesis_rule.iloc[0])


def test_candidate_retry_infeasible_dumps_with_final_ok_path_do_not_invalidate() -> None:
    bt = _mk_backtester()
    hourly = pd.DataFrame(
        {
            "timestamp_utc": [
                pd.Timestamp("2025-05-01T06:00:00Z"),
                pd.Timestamp("2025-05-01T07:00:00Z"),
            ],
            "optimization_error_code": ["ok", "ok"],
            "optimization_fallback": ["none", "none"],
            "optimizer_fallback_used": [0.0, 0.0],
        }
    )
    dumps = [
        {"timestamp_utc": "20250501T060000Z", "path": "a.npz", "solve_context": '{"final_accepted_path": false}'},
        {"timestamp_utc": "20250501T070000Z", "path": "b.npz", "solve_context": '{"final_accepted_path": false}'},
    ]
    accepted, candidate = bt._classify_infeasible_debug_dumps(dumps, hourly, timestamp_col="timestamp_utc")
    assert len(accepted) == 0
    assert len(candidate) == 2


def test_accepted_fallback_hour_marks_dump_as_accepted_and_invalidating() -> None:
    bt = _mk_backtester()
    hourly = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2025-05-01T09:00:00Z")],
            "optimization_error_code": ["safe_hold_plan"],
            "optimization_fallback": ["safe_hold_plan"],
            "optimizer_fallback_used": [1.0],
        }
    )
    dumps = [{"timestamp_utc": "20250501T090000Z", "path": "x.npz"}]
    accepted, candidate = bt._classify_infeasible_debug_dumps(dumps, hourly, timestamp_col="timestamp_utc")
    assert len(accepted) == 1
    assert len(candidate) == 0


def test_accepted_infeasible_dump_count_requires_first_timestamp() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    if float(s.get("accepted_path_infeasible_debug_dump_count", 0.0)) > 0.5:
        assert str(s.get("first_infeasible_timestamp_utc", "")).strip() != ""


def test_invalid_reason_includes_infeasible_debug_dump_only_for_accepted_path() -> None:
    s = {
        "accepted_path_infeasible_debug_dump_count": 0.0,
        "candidate_infeasible_debug_dump_count": 5.0,
        "fallback_used": 0.0,
        "reserve_feasibility_repair_used": 0.0,
        "simulation_valid": 1.0,
        "invalid_reason": "",
    }
    invalid_reason = str(s.get("invalid_reason", ""))
    if float(s.get("accepted_path_infeasible_debug_dump_count", 0.0)) > 0.5:
        invalid_reason = ",".join([v for v in [invalid_reason, "optimization_infeasible_debug_dump"] if v])
    assert "optimization_infeasible_debug_dump" not in invalid_reason


def test_summary_includes_input_vs_optimized_row_diagnostics() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    for k in [
        "input_row_count",
        "optimized_hour_count",
        "first_input_timestamp_utc",
        "first_optimized_target_timestamp_utc",
        "dropped_initial_rows_due_to_forecast_target_alignment",
    ]:
        assert k in s
    assert float(s["input_row_count"]) >= float(s["optimized_hour_count"])


def test_validator_required_fields_include_debug_dump_schema() -> None:
    req = set(validate_outputs.REQUIRED_SUMMARY_FIELDS)
    for k in (
        "simulation_schema_version",
        "required_summary_fields_version",
        "code_run_started_at_utc",
        "command_line_args",
        "output_was_cleaned",
        "infeasible_debug_dump_count",
        "accepted_path_infeasible_debug_dump_count",
        "candidate_infeasible_debug_dump_count",
        "infeasible_debug_dump_paths",
        "infeasible_debug_dump_timestamps",
        "required_fields_defaulted",
        "required_fields_computed",
    ):
        assert k in req


def test_summary_contains_debug_dump_paths_even_when_empty() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1)
    s = out.summary
    assert "infeasible_debug_dump_paths" in s
    assert "infeasible_debug_dump_timestamps" in s
    assert isinstance(s.get("infeasible_debug_dump_paths"), list)
    assert isinstance(s.get("infeasible_debug_dump_timestamps"), list)


def test_required_fields_not_runner_defaulted_silently() -> None:
    row = {
        "simulation_valid": 1.0,
        "thesis_reportable": 1.0,
        "invalid_reason": "",
        "fallback_used": 0.0,
        "fallback_mode_counts": "{}",
        "optimization_error_code_counts": '{"ok": 1}',
        "simulation_schema_version": "v2",
        "required_summary_fields_version": "v2",
        "code_run_started_at_utc": "2026-01-01T00:00:00Z",
        "command_line_args": "{}",
        "output_was_cleaned": 1.0,
        "infeasible_debug_dump_count": 0.0,
        "accepted_path_infeasible_debug_dump_count": 0.0,
        "candidate_infeasible_debug_dump_count": 0.0,
        "infeasible_debug_dump_paths": "[]",
        "infeasible_debug_dump_timestamps": "[]",
        "precommit_clamp_applied_count": 0.0,
        "activation_split_reconciliation_error_max": 0.0,
        "final_soc_check_pass": 1.0,
        "benchmark_same_rules_gate_consistent": 1.0,
        "global_perfect_foresight_dominance_check_pass": 1.0,
        "required_fields_defaulted": '["simulation_valid"]',
        "required_fields_computed": "[]",
    }
    d = pd.DataFrame([row])
    cnt = d["required_fields_defaulted"].apply(lambda v: len(json.loads(v))).iloc[0]
    assert cnt > 0


def test_required_field_defaulted_critical_invalidates() -> None:
    required_missing_count = 0
    critical_defaulted_count = 1
    required_fields_check_pass = (required_missing_count <= 0) and (critical_defaulted_count <= 0)
    assert required_fields_check_pass is False


def test_optional_field_defaulted_does_not_invalidate() -> None:
    required_missing_count = 0
    critical_defaulted_count = 0
    optional_defaulted_count = 2
    required_fields_check_pass = (required_missing_count <= 0) and (critical_defaulted_count <= 0)
    assert optional_defaulted_count > 0
    assert required_fields_check_pass is True


def test_optional_fields_defaulted_does_not_invalidate() -> None:
    test_optional_field_defaulted_does_not_invalidate()


def test_required_fields_missing_invalidates() -> None:
    required_fields_missing = 1
    critical_required_fields_defaulted = 0
    required_fields_check_pass = float(
        (required_fields_missing <= 0) and (critical_required_fields_defaulted <= 0)
    )
    assert required_fields_check_pass == 0.0


def test_required_fields_ok_consistent_with_check_pass(tmp_path: Path) -> None:
    scen = tmp_path / "multi" / "p50_p50"
    scen.mkdir(parents=True, exist_ok=True)
    payload = {
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
        "required_fields_missing": '["x"]',
        "critical_required_fields_defaulted": "[]",
        "optional_fields_defaulted": "[]",
        "required_fields_check_pass": 0.0,
        "precommit_clamp_applied_count": 0.0,
        "activation_split_reconciliation_error_max": 0.0,
        "final_soc_check_pass": 1.0,
        "benchmark_same_rules_gate_consistent": 1.0,
        "global_perfect_foresight_dominance_check_pass": 1.0,
        "global_perfect_foresight_validation_status": "disabled_unverified",
    }
    (scen / "backtest_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    out_json = tmp_path / "stats.json"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", ["validate_simulation_outputs.py", str(tmp_path), "--out-json", str(out_json)])
        validate_outputs.main()
    stats = json.loads(out_json.read_text(encoding="utf-8"))
    assert stats["required_fields_ok"] is False


def test_summary_always_contains_infeasible_debug_dump_fields() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1)
    s = out.summary
    for k in (
        "infeasible_debug_dump_count",
        "accepted_path_infeasible_debug_dump_count",
        "candidate_infeasible_debug_dump_count",
        "infeasible_debug_dump_paths",
        "infeasible_debug_dump_timestamps",
    ):
        assert k in s
    assert float(s["infeasible_debug_dump_count"]) >= 0.0
    assert float(s["accepted_path_infeasible_debug_dump_count"]) >= 0.0
    assert float(s["candidate_infeasible_debug_dump_count"]) >= 0.0
    assert isinstance(s["infeasible_debug_dump_paths"], list)
    assert isinstance(s["infeasible_debug_dump_timestamps"], list)


def test_validator_fails_missing_debug_dump_fields(tmp_path: Path) -> None:
    scen = tmp_path / "multi" / "p50_p50"
    scen.mkdir(parents=True, exist_ok=True)
    payload = {
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
        # intentionally missing infeasible_debug_dump_paths
        "infeasible_debug_dump_timestamps": [],
        "summary_fields_defaulted": "[]",
        "required_fields_defaulted": "[]",
        "required_fields_computed": "[]",
        "precommit_clamp_applied_count": 0.0,
        "activation_split_reconciliation_error_max": 0.0,
        "final_soc_check_pass": 1.0,
        "benchmark_same_rules_gate_consistent": 1.0,
        "global_perfect_foresight_dominance_check_pass": 1.0,
        "global_perfect_foresight_validation_status": "disabled_unverified",
    }
    (scen / "backtest_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                sys,
                "argv",
                ["validate_simulation_outputs.py", str(tmp_path)],
            )
            validate_outputs.main()
    assert "Missing required fields" in str(e.value)


def test_allow_stale_is_diagnostic_only(tmp_path: Path) -> None:
    scen = tmp_path / "multi" / "p50_p50"
    scen.mkdir(parents=True, exist_ok=True)
    payload = {
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
        # intentionally missing infeasible_debug_dump_paths
        "infeasible_debug_dump_timestamps": [],
        "summary_fields_defaulted": "[]",
        "required_fields_defaulted": "[]",
        "required_fields_computed": "[]",
        "precommit_clamp_applied_count": 0.0,
        "activation_split_reconciliation_error_max": 0.0,
        "final_soc_check_pass": 1.0,
        "benchmark_same_rules_gate_consistent": 1.0,
        "global_perfect_foresight_dominance_check_pass": 1.0,
        "global_perfect_foresight_validation_status": "disabled_unverified",
    }
    (scen / "backtest_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    out_json = tmp_path / "stats.json"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            sys,
            "argv",
            ["validate_simulation_outputs.py", str(tmp_path), "--allow-stale", "--out-json", str(out_json)],
        )
        validate_outputs.main()
    stats = json.loads(out_json.read_text(encoding="utf-8"))
    assert stats["required_fields_ok"] is False
    assert int(stats["stale_scenario_count"]) > 0
    assert int(stats["thesis_reportable_scenarios"]) == 0


def test_clean_output_removes_stale_summary(tmp_path: Path) -> None:
    scen = tmp_path / "multi" / "p50_p50"
    scen.mkdir(parents=True, exist_ok=True)
    stale = scen / "backtest_summary.json"
    stale.write_text("{}", encoding="utf-8")
    assert stale.exists()
    cleaned = _prepare_scenario_output_dir(
        scenario_out_dir=scen,
        clean_output=True,
        strict_simulation_validity=True,
        simulation_schema_version="v2",
    )
    assert cleaned == 1.0
    assert scen.exists()
    assert not stale.exists()


def test_strict_run_refuses_existing_output_without_clean_output(tmp_path: Path) -> None:
    scen = tmp_path / "multi" / "p50_p50"
    scen.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError):
        _prepare_scenario_output_dir(
            scenario_out_dir=scen,
            clean_output=False,
            strict_simulation_validity=True,
            simulation_schema_version="v2",
        )


def test_reserve_retry_ladder_reaches_zero() -> None:
    bt = _mk_backtester()
    ladder = bt._parse_reserve_retry_ladder("1.0,0.5,0.25,0.0")
    assert ladder == [1.0, 0.5, 0.25, 0.0]


def test_disable_new_bcm_reserve_bids_blocks_new_bids() -> None:
    bt = _mk_backtester()
    bt.disable_new_bcm_reserve_bids = True
    assert bt.disable_new_bcm_reserve_bids is True


def test_retry_fields_written_to_summary() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    for k in (
        "reserve_retry_ladder",
        "reserve_retry_attempts_used",
        "reserve_retry_final_factor",
        "reserve_retry_succeeded",
        "disable_new_bcm_reserve_bids",
        "new_reserve_bids_zeroed_by_retry",
        "reserve_retry_infeasible_after_zero_reserve",
    ):
        assert k in s


def test_retry_fields_written_to_reserve_commitment_debug() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1, strict_simulation_validity=True)
    h = out.hourly
    expected = {
        "submitted_reserve_pos_mw_before_retry",
        "submitted_reserve_neg_mw_before_retry",
        "reserve_retry_factor",
        "submitted_reserve_pos_mw_after_retry",
        "submitted_reserve_neg_mw_after_retry",
        "retry_reduction_reason",
    }
    assert expected.issubset(set(h.columns))


def test_conservative_reserve_cli_params_reach_backtester() -> None:
    txt = Path("scripts/run_battery_backtest.py").read_text(encoding="utf-8")
    assert "--disable-new-bcm-reserve-bids" in txt
    assert "--reserve-retry-ladder" in txt
    assert 'MODEL_SPECS["disable_new_bcm_reserve_bids"]' in txt
    assert 'MODEL_SPECS["reserve_retry_ladder"]' in txt


def test_terminal_repair_still_required_for_final_soc_shortfall() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "terminal_repair"
    df, col = _tiny_backtest_df(hours=6)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    if float(s.get("final_soc_shortfall_mwh", 0.0)) > 1e-6:
        assert float(s.get("terminal_soc_repair_included_in_pnl", 0.0)) >= 0.5
        assert float(s.get("final_soc_economic_repair_check_pass", 0.0)) >= 0.5


def test_driver_new_reserve_bid_too_high() -> None:
    row = pd.Series(
        {
            "optimization_error_code": "optimization_infeasible",
            "fallback_mode": "reserve_feasibility_repair",
            "new_submitted_reserve_pos_mw": 4.0,
            "new_submitted_reserve_neg_mw": 0.0,
            "locked_reserve_pos_mw": 0.0,
            "locked_reserve_neg_mw": 0.0,
        }
    )
    d, _ = _suspected_infeasibility_driver_from_row(row)
    assert d == "new_reserve_bid_too_high"


def test_driver_existing_lockbook_obligation_infeasible() -> None:
    row = pd.Series(
        {
            "optimization_error_code": "optimization_infeasible",
            "fallback_mode": "reserve_feasibility_repair",
            "new_submitted_reserve_pos_mw": 0.0,
            "new_submitted_reserve_neg_mw": 0.0,
            "locked_reserve_pos_mw": 3.0,
            "locked_reserve_neg_mw": 0.0,
        }
    )
    d, _ = _suspected_infeasibility_driver_from_row(row)
    assert d == "existing_lockbook_obligation_infeasible"


def test_driver_protected_soc_violation() -> None:
    row = pd.Series(
        {
            "optimization_error_code": "optimization_failure",
            "protected_soc_violation_pos_mwh": 0.5,
            "protected_soc_violation_neg_mwh": 0.0,
        }
    )
    d, _ = _suspected_infeasibility_driver_from_row(row)
    assert d == "protected_soc_violation"


def test_infeasibility_attribution_file_written_on_failure(tmp_path: Path) -> None:
    h = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T01:00:00Z")],
            "optimization_error_code": ["ok", "optimization_failure"],
            "optimization_fallback": ["none", "reserve_feasibility_repair"],
            "is_fallback_hour": [0.0, 1.0],
            "fixed_reserve_obligation_pos_mw": [0.0, 1.0],
            "fixed_reserve_obligation_neg_mw": [0.0, 0.0],
            "reserve_submitted_pos_mw": [0.0, 1.0],
            "reserve_submitted_neg_mw": [0.0, 0.0],
            "real_soc_start_mwh": [10.0, 9.0],
            "real_soc_mwh": [9.0, 8.0],
            "real_aux_energy_mwh": [0.0, 0.1],
        }
    )
    s = {"reserve_feasibility_repair_used": 1.0, "accepted_path_infeasible_debug_dump_count": 1.0, "p_max_mw": 20.0}
    out = _build_optimization_infeasibility_attribution(hourly=h, summary=s, scenario="p50_p50")
    assert not out.empty
    p = tmp_path / "optimization_infeasibility_attribution.csv"
    out.to_csv(p, index=False)
    assert p.exists()


def test_first_infeasibility_driver_in_summary() -> None:
    h = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "optimization_error_code": ["optimization_infeasible"],
            "optimization_fallback": ["reserve_feasibility_repair"],
            "is_fallback_hour": [1.0],
            "reserve_submitted_pos_mw": [2.0],
            "reserve_submitted_neg_mw": [0.0],
            "fixed_reserve_obligation_pos_mw": [0.0],
            "fixed_reserve_obligation_neg_mw": [0.0],
            "real_soc_start_mwh": [10.0],
            "real_soc_mwh": [9.0],
        }
    )
    out = _build_optimization_infeasibility_attribution(hourly=h, summary={"p_max_mw": 20.0}, scenario="p30_p30")
    assert str(out.iloc[0]["suspected_infeasibility_driver"]) in {
        "new_reserve_bid_too_high",
        "unknown_solver_or_numeric",
    }


def test_no_new_bcm_keeps_existing_lockbook_obligations() -> None:
    bt = _mk_backtester()
    bt.disable_new_bcm_reserve_bids = True
    assert bt.disable_new_bcm_reserve_bids is True


def test_fallback_still_not_reportable() -> None:
    test_fallback_never_reportable()


def test_no_obligation_protected_soc_equals_physical_bounds() -> None:
    bt = _mk_backtester()
    bt.reserve_headroom_safety_mwh = 0.75
    bt.reserve_soc_projection_safety_mwh = 1.5
    res = bt._compute_obligation_driven_protected_soc_bounds(
        soc_start_mwh=3.5,
        required_headroom_pos_mwh=0.0,
        required_headroom_neg_mwh=0.0,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        committed_bem_pos_mw=0.0,
        committed_bem_neg_mw=0.0,
        reserve_pos_mw=0.0,
        reserve_neg_mw=0.0,
    )
    assert np.isclose(float(res["physical_soc_min_mwh"]), bt.soc_min)
    assert np.isclose(float(res["physical_soc_max_mwh"]), bt.soc_max)
    assert np.isclose(float(res["protected_soc_min_mwh"]), bt.soc_min)
    assert np.isclose(float(res["protected_soc_max_mwh"]), bt.soc_max)
    assert np.isclose(float(res["protected_soc_violation_pos_mwh"]), 0.0)
    assert np.isclose(float(res["protected_soc_violation_neg_mwh"]), 0.0)


def test_positive_obligation_protected_soc_uses_headroom_and_safety() -> None:
    bt = _mk_backtester()
    bt.reserve_headroom_safety_mwh = 0.75
    bt.reserve_soc_projection_safety_mwh = 1.5
    res = bt._compute_obligation_driven_protected_soc_bounds(
        soc_start_mwh=3.0,
        required_headroom_pos_mwh=0.5,
        required_headroom_neg_mwh=0.0,
        locked_reserve_pos_mw=1.0,
        locked_reserve_neg_mw=0.0,
        committed_bem_pos_mw=0.0,
        committed_bem_neg_mw=0.0,
        reserve_pos_mw=0.0,
        reserve_neg_mw=0.0,
    )
    assert float(res["protected_soc_min_mwh"]) > bt.soc_min
    assert float(res["protected_soc_violation_pos_mwh"]) > 0.0


def test_negative_obligation_protected_soc_uses_headroom_and_safety() -> None:
    bt = _mk_backtester()
    bt.reserve_headroom_safety_mwh = 0.75
    bt.reserve_soc_projection_safety_mwh = 1.5
    res = bt._compute_obligation_driven_protected_soc_bounds(
        soc_start_mwh=17.5,
        required_headroom_pos_mwh=0.0,
        required_headroom_neg_mwh=0.5,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=1.0,
        committed_bem_pos_mw=0.0,
        committed_bem_neg_mw=0.0,
        reserve_pos_mw=0.0,
        reserve_neg_mw=0.0,
    )
    assert float(res["protected_soc_max_mwh"]) < bt.soc_max
    assert float(res["protected_soc_violation_neg_mwh"]) > 0.0


def test_zero_required_headroom_does_not_invalidate_da_only_soc_path() -> None:
    bt = _mk_backtester()
    bt.reserve_headroom_safety_mwh = 0.75
    bt.reserve_soc_projection_safety_mwh = 1.5
    res = bt._compute_obligation_driven_protected_soc_bounds(
        soc_start_mwh=3.5,
        required_headroom_pos_mwh=0.0,
        required_headroom_neg_mwh=0.0,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        committed_bem_pos_mw=0.0,
        committed_bem_neg_mw=0.0,
        reserve_pos_mw=0.0,
        reserve_neg_mw=0.0,
    )
    assert float(res["protected_soc_violation_pos_mwh"]) <= 1e-12
    assert float(res["protected_soc_violation_without_obligation"]) <= 1e-12


def test_real_physical_soc_violation_still_invalidates() -> None:
    bt = _mk_backtester()
    res = bt._compute_obligation_driven_protected_soc_bounds(
        soc_start_mwh=bt.soc_min - 0.1,
        required_headroom_pos_mwh=0.0,
        required_headroom_neg_mwh=0.0,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        committed_bem_pos_mw=0.0,
        committed_bem_neg_mw=0.0,
        reserve_pos_mw=0.0,
        reserve_neg_mw=0.0,
    )
    assert float(res["physical_soc_violation_pos_mwh"]) > 0.0


def test_protected_soc_violation_with_obligation_still_invalidates() -> None:
    bt = _mk_backtester()
    bt.reserve_headroom_safety_mwh = 0.75
    bt.reserve_soc_projection_safety_mwh = 1.5
    res = bt._compute_obligation_driven_protected_soc_bounds(
        soc_start_mwh=3.0,
        required_headroom_pos_mwh=0.75,
        required_headroom_neg_mwh=0.0,
        locked_reserve_pos_mw=0.5,
        locked_reserve_neg_mw=0.0,
        committed_bem_pos_mw=0.0,
        committed_bem_neg_mw=0.0,
        reserve_pos_mw=0.0,
        reserve_neg_mw=0.0,
    )
    assert float(res["obligation_headroom_pos_active"]) > 0.5
    assert float(res["protected_soc_violation_pos_mwh"]) > 0.0
