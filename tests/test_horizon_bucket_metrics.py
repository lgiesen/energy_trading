import math

import numpy as np
import pandas as pd
import pytest

from energy_trading.evaluation.metrics import compute_horizon_bucket_metrics


def _mk_times(snapshot_local_hour: int, lead: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    snap_local = pd.Timestamp("2026-01-01", tz="Europe/Berlin") + pd.Timedelta(hours=int(snapshot_local_hour))
    snap_utc = snap_local.tz_convert("UTC")
    target_utc = snap_utc + pd.Timedelta(hours=int(lead))
    return snap_utc, target_utc


def test_activation_buckets_partition():
    rows = []
    truth_rows = []
    for lead in [1, 8, 9, 16, 17, 24]:
        snap, tgt = _mk_times(5, lead)
        rows.append(
            {
                "snapshot_time_utc": snap,
                "target_time_utc": tgt,
                "lead_time_h": lead,
                "p50": 10.0,
                "p10": 9.0,
                "p90": 11.0,
            }
        )
        truth_rows.append({"timestamp_utc": tgt, "target_afrr_activation_price_vwap_pos": 10.0})
    out = compute_horizon_bucket_metrics(
        pd.DataFrame(rows),
        pd.DataFrame(truth_rows),
        target_col="target_afrr_activation_price_vwap_pos",
        y_pred_col="p50",
    )
    assert out["horizon_short_h1_8_n"] == 2
    assert out["horizon_medium_h9_16_n"] == 2
    assert out["horizon_long_h17_max_n"] == 2


def test_capacity_actionable_bucket_filters():
    rows = []
    truth_rows = []
    for hour, lead in [(8, 16), (8, 39), (7, 16), (9, 16), (8, 15), (8, 40)]:
        snap, tgt = _mk_times(hour, lead)
        rows.append(
            {
                "snapshot_time_utc": snap,
                "target_time_utc": tgt,
                "lead_time_h": lead,
                "p50": 10.0,
            }
        )
        truth_rows.append({"timestamp_utc": tgt, "target_afrr_capacity_price_pos": 10.0})
    out = compute_horizon_bucket_metrics(
        pd.DataFrame(rows),
        pd.DataFrame(truth_rows),
        target_col="target_afrr_capacity_price_pos",
        y_pred_col="p50",
        max_horizon=48,
    )
    assert out["horizon_actionable_capacity_gate_dplus1_n"] == 2


def test_da_actionable_bucket_filters():
    rows = []
    truth_rows = []
    for hour, lead in [(12, 12), (12, 35), (11, 12), (13, 12), (12, 11), (12, 36)]:
        snap, tgt = _mk_times(hour, lead)
        rows.append(
            {
                "snapshot_time_utc": snap,
                "target_time_utc": tgt,
                "lead_time_h": lead,
                "p50": 20.0,
            }
        )
        truth_rows.append({"timestamp_utc": tgt, "target_da_price": 20.0})
    out = compute_horizon_bucket_metrics(
        pd.DataFrame(rows),
        pd.DataFrame(truth_rows),
        target_col="target_da_price",
        y_pred_col="p50",
        max_horizon=48,
    )
    assert out["horizon_actionable_da_gate_dplus1_n"] == 2


def test_empty_bucket_returns_nan():
    rows = []
    truth_rows = []
    for lead in [1, 2, 3]:
        snap, tgt = _mk_times(4, lead)
        rows.append({"snapshot_time_utc": snap, "target_time_utc": tgt, "lead_time_h": lead, "p50": 1.0})
        truth_rows.append({"timestamp_utc": tgt, "target_afrr_activation_rate_pos": 1.0})
    out = compute_horizon_bucket_metrics(
        pd.DataFrame(rows),
        pd.DataFrame(truth_rows),
        target_col="target_afrr_activation_rate_pos",
        y_pred_col="p50",
        max_horizon=16,
    )
    assert out["horizon_long_h17_max_n"] == 0
    assert math.isnan(float(out["horizon_long_h17_max_mae"]))


def test_unknown_target_raises():
    with pytest.raises(ValueError):
        compute_horizon_bucket_metrics(
            pd.DataFrame(
                [
                    {
                        "snapshot_time_utc": pd.Timestamp("2026-01-01T00:00:00Z"),
                        "target_time_utc": pd.Timestamp("2026-01-01T01:00:00Z"),
                        "lead_time_h": 1,
                        "p50": 1.0,
                    }
                ]
            ),
            pd.DataFrame([{"timestamp_utc": pd.Timestamp("2026-01-01T01:00:00Z"), "typo_target": 1.0}]),
            target_col="typo_target",
            y_pred_col="p50",
        )


def test_bucket_metrics_are_unweighted_mean():
    snap1, tgt1 = _mk_times(12, 12)
    snap2, tgt2 = _mk_times(12, 13)
    pred = pd.DataFrame(
        [
            {"snapshot_time_utc": snap1, "target_time_utc": tgt1, "lead_time_h": 12, "p50": 0.0},
            {"snapshot_time_utc": snap2, "target_time_utc": tgt2, "lead_time_h": 13, "p50": 0.0},
        ]
    )
    truth = pd.DataFrame(
        [
            {"timestamp_utc": tgt1, "target_da_price": 0.0},
            {"timestamp_utc": tgt2, "target_da_price": 10.0},
        ]
    )
    out = compute_horizon_bucket_metrics(
        pred,
        truth,
        target_col="target_da_price",
        y_pred_col="p50",
        max_horizon=48,
    )
    assert out["horizon_actionable_da_gate_dplus1_n"] == 2
    assert out["horizon_actionable_da_gate_dplus1_mae"] == pytest.approx(5.0)

