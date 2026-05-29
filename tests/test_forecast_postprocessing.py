from __future__ import annotations

from pathlib import Path
import subprocess
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

from energy_trading.evaluation.forecast_postprocessing import (  # noqa: E402
    canonicalize_prediction_frame,
    canonicalize_truth_series,
)


def test_negative_capacity_not_transformed() -> None:
    df = pd.DataFrame({"p10": [-100.0], "p30": [-80.0], "p50": [-50.0], "p70": [-20.0], "p90": [-10.0]})
    out, _ = canonicalize_prediction_frame(df, "pred_afrr_capacity_price_neg", ["p10", "p30", "p50", "p70", "p90"])
    assert np.isclose(float(out.loc[0, "p10"]), -100.0)
    assert np.isclose(float(out.loc[0, "p30"]), -80.0)
    assert np.isclose(float(out.loc[0, "p50"]), -50.0)
    assert np.isclose(float(out.loc[0, "p70"]), -20.0)
    assert np.isclose(float(out.loc[0, "p90"]), -10.0)


def test_negative_activation_quantile_flip_exact() -> None:
    df = pd.DataFrame({"p10": [-1000.0], "p30": [-700.0], "p50": [-500.0], "p70": [-300.0], "p90": [-100.0]})
    out, _ = canonicalize_prediction_frame(df, "pred_afrr_activation_price_neg", ["p10", "p30", "p50", "p70", "p90"])
    assert np.isclose(float(out.loc[0, "p90"]), 1000.0)
    assert np.isclose(float(out.loc[0, "p10"]), 100.0)


def test_predicted_value_sign_flipped_once_and_no_p50_double_flip() -> None:
    df = pd.DataFrame({
        "predicted_value": [-50.0],
        "p10": [-100.0],
        "p30": [-80.0],
        "p50": [-50.0],
        "p70": [-20.0],
        "p90": [-10.0],
    })
    out, _ = canonicalize_prediction_frame(df, "pred_afrr_activation_price_neg", ["p10", "p30", "p50", "p70", "p90"])
    assert np.isclose(float(out.loc[0, "predicted_value"]), 50.0)
    assert np.isclose(float(out.loc[0, "p50"]), 50.0)


def test_activation_rates_clipped_to_unit_interval() -> None:
    df = pd.DataFrame({"p10": [-0.2], "p50": [0.5], "p90": [1.3], "predicted_value": [2.0]})
    out, _ = canonicalize_prediction_frame(df, "pred_afrr_activation_rate_neg", ["p10", "p50", "p90"])
    assert np.isclose(float(out.loc[0, "p10"]), 0.0)
    assert np.isclose(float(out.loc[0, "p50"]), 0.5)
    assert np.isclose(float(out.loc[0, "p90"]), 1.0)
    assert np.isclose(float(out.loc[0, "predicted_value"]), 1.0)


def test_capacity_values_clipped_lower_zero() -> None:
    s = pd.Series([-10.0, 5.0])
    out = canonicalize_truth_series(s, "pred_afrr_capacity_price_pos")
    assert np.isclose(float(out.iloc[0]), 0.0)
    assert np.isclose(float(out.iloc[1]), 5.0)


def test_da_price_not_clipped_and_not_flipped() -> None:
    s = pd.Series([-50.0, 10.0])
    out = canonicalize_truth_series(s, "pred_da_price")
    assert np.isclose(float(out.iloc[0]), -50.0)
    assert np.isclose(float(out.iloc[1]), 10.0)


def test_benchmark_truth_and_prediction_both_canonicalized() -> None:
    s_true = pd.Series([-100.0])
    s_pred = pd.Series([-80.0])
    true_c = canonicalize_truth_series(s_true, "pred_afrr_activation_price_neg")
    pred_c = canonicalize_truth_series(s_pred, "pred_afrr_activation_price_neg")
    assert np.isclose(float(true_c.iloc[0]), 100.0)
    assert np.isclose(float(pred_c.iloc[0]), 80.0)


def test_missing_required_quantile_pair_fails() -> None:
    df = pd.DataFrame({"p10": [-100.0], "p50": [-50.0], "p70": [-20.0], "p90": [-10.0]})
    with pytest.raises(ValueError):
        canonicalize_prediction_frame(df, "pred_afrr_activation_price_neg", ["p10", "p50", "p70", "p90"])


def test_canonical_activation_neg_artifact_does_not_flip_quantiles() -> None:
    df = pd.DataFrame({"p10": [10.0], "p30": [20.0], "p50": [50.0], "p70": [80.0], "p90": [100.0]})
    out, _ = canonicalize_prediction_frame(
        df,
        "pred_afrr_activation_price_neg",
        ["p10", "p30", "p50", "p70", "p90"],
        target_value_mode="canonical_economic",
    )
    assert np.isclose(float(out.loc[0, "p10"]), 10.0)
    assert np.isclose(float(out.loc[0, "p90"]), 100.0)


def test_legacy_raw_activation_neg_artifact_flips_quantiles() -> None:
    df = pd.DataFrame({"p10": [-100.0], "p30": [-80.0], "p50": [-50.0], "p70": [-20.0], "p90": [-10.0]})
    out, _ = canonicalize_prediction_frame(
        df,
        "pred_afrr_activation_price_neg",
        ["p10", "p30", "p50", "p70", "p90"],
        target_value_mode="raw_signed_legacy",
    )
    assert np.isclose(float(out.loc[0, "p10"]), 10.0)
    assert np.isclose(float(out.loc[0, "p90"]), 100.0)


def test_check_forecast_postprocessing_self_test_passes() -> None:
    cp = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "check_forecast_postprocessing.py"), "--self-test"],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
