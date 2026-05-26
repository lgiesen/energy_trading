import math

import numpy as np
import pytest

from energy_trading.models.horizon_weighting import get_lead_sample_weights, get_training_weights


def _high_leads(weights: np.ndarray) -> list[int]:
    idx = np.where(np.isclose(weights, 1.0))[0]
    return [int(i + 1) for i in idx]


def test_capacity_last_hour_only_dplus1():
    w = get_training_weights("target_afrr_capacity_price_pos", current_hour=8, horizon=48)
    assert w.shape == (48,)
    assert _high_leads(w) == list(range(16, 40))
    baseline_idx = [0, 14, 39, 47]
    for i in baseline_idx:
        assert w[i] == pytest.approx(0.05)


def test_capacity_non_gate_all_baseline():
    for h in [7, 13]:
        w = get_training_weights("target_afrr_capacity_price_pos", current_hour=h, horizon=48)
        assert np.allclose(w, 0.05)
        assert not np.allclose(w, 1.0)


def test_da_last_hour_only_dplus1():
    w = get_training_weights("target_da_price", current_hour=12, horizon=48)
    assert w.shape == (48,)
    assert _high_leads(w) == list(range(12, 36))
    for i in [0, 10, 35, 47]:
        assert w[i] == pytest.approx(0.05)


def test_da_non_gate_all_baseline():
    for h in [11, 13]:
        w = get_training_weights("target_da_price", current_hour=h, horizon=48)
        assert np.allclose(w, 0.05)
        assert not np.allclose(w, 1.0)


def test_no_dplus2_shift():
    w_cap = get_training_weights("target_afrr_capacity_price_pos", current_hour=9, horizon=72)
    w_da = get_training_weights("target_da_price", current_hour=13, horizon=72)
    assert not np.any(np.isclose(w_cap[48:72], 1.0))
    assert not np.any(np.isclose(w_da[48:72], 1.0))


def test_activation_decay():
    w = get_training_weights("target_afrr_activation_price_vwap_pos", current_hour=3, horizon=72)
    assert w[0] == pytest.approx(1.0)
    assert w[6] == pytest.approx(0.5, rel=1e-12, abs=1e-12)
    assert np.min(w) >= 0.05
    assert w[20] == pytest.approx(max(0.05, math.exp(-math.log(2.0) * 20 / 6.0)))


def test_get_lead_sample_weights_direct():
    assert get_lead_sample_weights("target_afrr_capacity_price_pos", [8], 16)[0] == pytest.approx(1.0)
    assert get_lead_sample_weights("target_afrr_capacity_price_pos", [8], 15)[0] == pytest.approx(0.05)
    assert get_lead_sample_weights("target_afrr_capacity_price_pos", [7], 16)[0] == pytest.approx(0.05)
    assert get_lead_sample_weights("target_da_price", [12], 12)[0] == pytest.approx(1.0)
    assert get_lead_sample_weights("target_da_price", [12], 36)[0] == pytest.approx(0.05)


def test_unknown_target_raises():
    with pytest.raises(ValueError):
        get_training_weights("typo_target", 8, 48)
    with pytest.raises(ValueError):
        get_lead_sample_weights("typo_target", [8], 16)

