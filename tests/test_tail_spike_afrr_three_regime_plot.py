from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_final_tail_spike_benchmark import (
    ModelSpec,
    afrr_three_regime_plot_source,
    build_afrr_main_regime_outputs,
)


MODELS = [ModelSpec("tft", "TFT"), ModelSpec("xgb", "XGB"), ModelSpec("linear", "RLQR")]


def _target_frame(target: str) -> pd.DataFrame:
    rows = []
    for i, (ts, y) in enumerate(
        [
            ("2025-01-06 00:00", 100.0),  # stress week and high-tail; high-tail must win
            ("2025-01-07 00:00", 20.0),  # stress only
            ("2025-01-14 00:00", 10.0),  # non-stress
        ]
    ):
        rows.append(
            {
                "forecast_time_utc": pd.Timestamp(ts, tz="UTC") - pd.Timedelta(hours=24),
                "target_time_utc": pd.Timestamp(ts, tz="UTC"),
                "lead_time_h": 24.0,
                "target": target,
                "y_true": y,
                "p50": y,
            }
        )
    return pd.DataFrame(rows)


def _loaded() -> tuple[dict[tuple[str, str, str], pd.DataFrame], dict[tuple[str, str, str], dict[float, str]]]:
    targets = ["target_afrr_activation_rate_pos", "target_afrr_activation_rate_neg"]
    loaded = {}
    qmaps = {}
    for target in targets:
        base = _target_frame(target)
        for model, offset in [("linear", 2.0), ("xgb", 1.0), ("tft", 3.0)]:
            df = base.copy()
            df["p50"] = df["y_true"] + offset
            loaded[(model, "test", target)] = df
            qmaps[(model, "test", target)] = {0.5: "p50"}
    return loaded, qmaps


def test_afrr_three_regime_assignment_and_relative_pinball() -> None:
    loaded, qmaps = _loaded()
    selected_weeks = pd.DataFrame(
        [
            {
                "split": "test",
                "target": target,
                "target_display": target,
                "target_group": "aFRR activation rate",
                "selection_type": "high_volatility_week",
                "week_start_utc": pd.Timestamp("2025-01-06", tz="UTC"),
                "volatility_score": 1.0,
                "n_rows": 2,
            }
            for target in ["target_afrr_activation_rate_pos", "target_afrr_activation_rate_neg"]
        ]
    )
    warnings: list[dict] = []

    outputs = build_afrr_main_regime_outputs(
        loaded=loaded,
        qmaps=qmaps,
        models=MODELS,
        splits=["test"],
        selected_weeks=selected_weeks,
        epsilon=1e-9,
        warnings=warnings,
        eval_origin_start=None,
        eval_origin_end=None,
    )
    source = afrr_three_regime_plot_source(outputs["metrics"], split="test", warnings=warnings)

    assert set(source["regime_display"]) == {"Non-stress regime", "Stress regime", "High tail 5%"}
    assert not source["regime_internal"].isin(["activation_nonzero", "activation_zero_or_nearzero"]).any()
    assert set(source["target"]) == {"target_afrr_activation_rate_pos", "target_afrr_activation_rate_neg"}
    assert source["mutually_exclusive_assignment"].eq(True).all()

    points = outputs["points"]
    assigned = points.groupby(["target", "model", "target_time_utc"])["regime_display"].nunique()
    assert int(assigned.max()) == 1
    high_tail_rows = points.loc[pd.to_datetime(points["target_time_utc"], utc=True).eq(pd.Timestamp("2025-01-06", tz="UTC"))]
    assert set(high_tail_rows["regime_display"]) == {"High tail 5%"}
    stress_rows = points.loc[pd.to_datetime(points["target_time_utc"], utc=True).eq(pd.Timestamp("2025-01-07", tz="UTC"))]
    assert set(stress_rows["regime_display"]) == {"Stress regime"}
    baseline_rows = points.loc[pd.to_datetime(points["target_time_utc"], utc=True).eq(pd.Timestamp("2025-01-14", tz="UTC"))]
    assert set(baseline_rows["regime_display"]) == {"Non-stress regime"}

    xgb_row = source.loc[
        source["model_label"].eq("XGB")
        & source["target"].eq("target_afrr_activation_rate_pos")
        & source["regime_display"].eq("High tail 5%")
    ].iloc[0]
    assert xgb_row["relative_pinball_loss"] == pytest.approx(0.5)

