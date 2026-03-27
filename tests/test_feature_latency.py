from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import polars as pl


# Make sure local src/ is importable when running pytest from repo root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_trading.features.build_features import (  # noqa: E402
    add_multi_output_targets,
    apply_point_in_time_lag_layer,
)


def _build_dummy_df(n_rows: int = 200) -> pl.DataFrame:
    ts = pl.datetime_range(
        start=pl.datetime(2024, 1, 1, 0, 0, 0, time_zone="UTC"),
        end=pl.datetime(2024, 1, 1, 0, 0, 0, time_zone="UTC") + (n_rows - 1) * pl.duration(hours=1),
        interval="1h",
        eager=True,
    )
    vals = np.arange(1, n_rows + 1, dtype=float)
    return pl.DataFrame(
        {
            "timestamp_utc": ts,
            "load_actual_entsoe": vals,  # 2h lag group
            "afrr_activated_mw_pos": vals,  # 1h lag group
            "da_price": vals,  # 0h lag group
            "y_true_pos": vals,  # target
            "y_true_neg": vals,  # required by add_multi_output_targets in pipeline
        }
    )


def test_strict_lag_first_integrity() -> None:
    df_raw = _build_dummy_df(200)
    raw_load = df_raw["load_actual_entsoe"].to_numpy()
    raw_afrr = df_raw["afrr_activated_mw_pos"].to_numpy()
    raw_da = df_raw["da_price"].to_numpy()
    raw_y = df_raw["y_true_pos"].to_numpy()

    # Run the lag layer exactly as in build_features.
    df_lagged = apply_point_in_time_lag_layer(df_raw)

    # Add sequence targets exactly as in build_features.
    df_out = add_multi_output_targets(df_lagged, horizon_hours=72)

    # Use i > 170 as requested for lag checks.
    i = 180
    got_load = float(df_out.row(i, named=True)["load_actual_entsoe"])
    got_afrr = float(df_out.row(i, named=True)["afrr_activated_mw_pos"])
    got_da = float(df_out.row(i, named=True)["da_price"])

    assert got_load == raw_load[i - 2], "2-hour lag mismatch for load_actual_entsoe"
    assert got_afrr == raw_afrr[i - 1], "1-hour lag mismatch for afrr_activated_mw_pos"
    assert got_da == raw_da[i], "0-hour lag mismatch for da_price"
    print("PASSED: Lag mapping (2h / 1h / 0h) is correct.")

    # Rolling integrity: 24h rolling mean on lagged load must not use raw i or i-1.
    lagged_load = df_out["load_actual_entsoe"].to_numpy()
    rolling_24h_at_i = float(np.mean(lagged_load[i - 23 : i + 1]))
    expected_from_raw = float(np.mean(raw_load[i - 25 : i - 1]))  # raw i-25 ... i-2
    assert np.isclose(rolling_24h_at_i, expected_from_raw), "24h rolling mean includes non-causal values"
    assert not np.isclose(rolling_24h_at_i, np.mean(raw_load[i - 23 : i + 1])), "Rolling mean leaked raw current/future"
    print("PASSED: 24h rolling window is causal and uses lagged inputs only.")

    # Target alignment must use a row where i+72 exists.
    i_target = 100
    got_h1 = float(df_out.row(i_target, named=True)["target_pos_h1"])
    got_h72 = float(df_out.row(i_target, named=True)["target_pos_h72"])
    assert got_h1 == raw_y[i_target + 1], "target_pos_h1 misalignment"
    assert got_h72 == raw_y[i_target + 72], "target_pos_h72 misalignment"
    print("PASSED: Multi-output target alignment (h1..h72) is correct.")

    # Guardrail: y_true_pos itself must not be lagged.
    got_y_now = float(df_out.row(i, named=True)["y_true_pos"])
    assert got_y_now == raw_y[i], "y_true_pos was unexpectedly lagged"
    print("PASSED: Ground-truth target y_true_pos remains unlagged.")
