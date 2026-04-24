"""Unit tests for payoff functions and the dispatch layer.

Covers:
  - call/put payoffs and put-call parity
  - call spread and put spread payoffs
  - straddle payoff
  - dispatch layer: price consistency, delta bounds, delta transform ranges
"""

from __future__ import annotations

import numpy as np
import pytest

from src.payoffs.european import call_payoff, put_payoff, call_price, put_price, call_delta, put_delta
from src.payoffs.spreads import call_spread_payoff, put_spread_payoff
from src.payoffs.straddles import straddle_payoff
from src.payoffs.dispatch import get_payoff_spec, PayoffSpec
from src.experiments.configs import PayoffConfig, MarketConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MC = MarketConfig()  # s0=100, k=100, r=0.02, sigma=0.20, T=0.5


def _mc() -> MarketConfig:
    return MC


# ---------------------------------------------------------------------------
# European call / put payoffs
# ---------------------------------------------------------------------------


def test_call_payoff_atm() -> None:
    assert float(call_payoff(100.0, 100.0)) == pytest.approx(0.0)


def test_call_payoff_itm() -> None:
    assert float(call_payoff(120.0, 100.0)) == pytest.approx(20.0)


def test_call_payoff_otm() -> None:
    assert float(call_payoff(80.0, 100.0)) == pytest.approx(0.0)


def test_put_payoff_atm() -> None:
    assert float(put_payoff(100.0, 100.0)) == pytest.approx(0.0)


def test_put_payoff_itm() -> None:
    assert float(put_payoff(80.0, 100.0)) == pytest.approx(20.0)


def test_put_payoff_otm() -> None:
    assert float(put_payoff(120.0, 100.0)) == pytest.approx(0.0)


def test_put_call_parity_payoffs() -> None:
    """call_payoff - put_payoff = S_T - K for all S_T."""
    s = np.linspace(50, 150, 50)
    k = 100.0
    np.testing.assert_allclose(
        call_payoff(s, k) - put_payoff(s, k), s - k, atol=1e-10
    )


# ---------------------------------------------------------------------------
# Call spread payoffs
# ---------------------------------------------------------------------------


def test_call_spread_below_lower_strike() -> None:
    assert float(call_spread_payoff(80.0, 95.0, 105.0)) == pytest.approx(0.0)


def test_call_spread_between_strikes() -> None:
    assert float(call_spread_payoff(100.0, 95.0, 105.0)) == pytest.approx(5.0)


def test_call_spread_above_upper_strike() -> None:
    assert float(call_spread_payoff(120.0, 95.0, 105.0)) == pytest.approx(10.0)


def test_call_spread_nonnegative() -> None:
    s = np.linspace(50, 200, 100)
    p = call_spread_payoff(s, 95.0, 105.0)
    assert np.all(p >= 0)


def test_call_spread_bounded_by_width() -> None:
    s = np.linspace(50, 200, 100)
    p = call_spread_payoff(s, 95.0, 105.0)
    assert np.all(p <= 10.0 + 1e-10)


def test_put_spread_mirror_call_spread() -> None:
    """put_spread(k_lo, k_hi) should mirror call_spread at the same strikes."""
    s = np.array([80.0, 95.0, 100.0, 105.0, 120.0])
    k_lo, k_hi = 95.0, 105.0
    cs = call_spread_payoff(s, k_lo, k_hi)
    ps = put_spread_payoff(s, k_lo, k_hi)
    # put spread = k_hi - k_lo - call_spread + (call_payoff(k_lo) - put_payoff(k_hi)) ... no.
    # Direct test: call + put spread = k_hi - k_lo for any S_T (like parity)
    np.testing.assert_allclose(cs + ps, np.full_like(s, k_hi - k_lo), atol=1e-10)


# ---------------------------------------------------------------------------
# Straddle payoffs
# ---------------------------------------------------------------------------


def test_straddle_atm() -> None:
    assert float(straddle_payoff(100.0, 100.0)) == pytest.approx(0.0)


def test_straddle_above() -> None:
    assert float(straddle_payoff(115.0, 100.0)) == pytest.approx(15.0)


def test_straddle_below() -> None:
    assert float(straddle_payoff(85.0, 100.0)) == pytest.approx(15.0)


def test_straddle_equals_abs() -> None:
    s = np.linspace(50, 150, 100)
    np.testing.assert_allclose(straddle_payoff(s, 100.0), np.abs(s - 100.0), atol=1e-10)


def test_straddle_equals_call_plus_put() -> None:
    s = np.linspace(50, 150, 100)
    k = 100.0
    np.testing.assert_allclose(
        straddle_payoff(s, k),
        call_payoff(s, k) + put_payoff(s, k),
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# PayoffSpec dispatch: prices
# ---------------------------------------------------------------------------


def test_dispatch_call_price_positive() -> None:
    spec = get_payoff_spec(PayoffConfig(payoff_type="call"), _mc())
    assert spec.initial_option_price > 0


def test_dispatch_put_price_positive() -> None:
    spec = get_payoff_spec(PayoffConfig(payoff_type="put"), _mc())
    assert spec.initial_option_price > 0


def test_dispatch_call_spread_price_less_than_call() -> None:
    mc = _mc()
    call_spec = get_payoff_spec(PayoffConfig(payoff_type="call"), mc)
    spread_spec = get_payoff_spec(PayoffConfig(payoff_type="call_spread", k=95.0, k_hi=105.0), mc)
    assert spread_spec.initial_option_price < call_spec.initial_option_price


def test_dispatch_straddle_price_equals_call_plus_put() -> None:
    mc = _mc()
    call_spec = get_payoff_spec(PayoffConfig(payoff_type="call"), mc)
    put_spec = get_payoff_spec(PayoffConfig(payoff_type="put"), mc)
    straddle_spec = get_payoff_spec(PayoffConfig(payoff_type="straddle"), mc)
    np.testing.assert_allclose(
        straddle_spec.initial_option_price,
        call_spec.initial_option_price + put_spec.initial_option_price,
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# PayoffSpec dispatch: classical deltas
# ---------------------------------------------------------------------------


def test_dispatch_call_delta_in_unit_interval() -> None:
    spec = get_payoff_spec(PayoffConfig(payoff_type="call"), _mc())
    s = np.linspace(60, 160, 40)
    d = spec.classical_delta_fn(s, 0.25)
    assert np.all(d >= 0) and np.all(d <= 1)


def test_dispatch_put_delta_negative() -> None:
    spec = get_payoff_spec(PayoffConfig(payoff_type="put"), _mc())
    s = np.linspace(60, 160, 40)
    d = spec.classical_delta_fn(s, 0.25)
    assert np.all(d >= -1) and np.all(d <= 0)


def test_dispatch_put_delta_equals_call_minus_one() -> None:
    mc = _mc()
    call_spec = get_payoff_spec(PayoffConfig(payoff_type="call"), mc)
    put_spec = get_payoff_spec(PayoffConfig(payoff_type="put"), mc)
    s = np.linspace(60, 160, 40)
    tau = 0.25
    np.testing.assert_allclose(
        put_spec.classical_delta_fn(s, tau),
        call_spec.classical_delta_fn(s, tau) - 1.0,
        atol=1e-10,
    )


def test_dispatch_call_spread_delta_in_unit_interval() -> None:
    spec = get_payoff_spec(PayoffConfig(payoff_type="call_spread", k=95.0, k_hi=105.0), _mc())
    s = np.linspace(60, 160, 40)
    d = spec.classical_delta_fn(s, 0.25)
    assert np.all(d >= -1e-9) and np.all(d <= 1 + 1e-9)


def test_dispatch_straddle_delta_symmetric_atm() -> None:
    spec = get_payoff_spec(PayoffConfig(payoff_type="straddle"), _mc())
    # ATM delta should be ≈ 0
    d_atm = spec.classical_delta_fn(np.array([100.0]), np.array([0.25]))
    assert abs(float(d_atm.flat[0])) < 0.1  # near zero ATM


# ---------------------------------------------------------------------------
# PayoffSpec dispatch: delta transforms
# ---------------------------------------------------------------------------


def test_delta_transform_call_identity() -> None:
    spec = get_payoff_spec(PayoffConfig(payoff_type="call"), _mc())
    import torch
    x = torch.tensor([0.3, 0.5, 0.8])
    result = spec.delta_transform(x)
    np.testing.assert_allclose(result.numpy(), x.numpy(), atol=1e-7)


def test_delta_transform_put_shifts_to_negative() -> None:
    spec = get_payoff_spec(PayoffConfig(payoff_type="put"), _mc())
    import torch
    x = torch.tensor([0.3, 0.5, 0.8])
    result = spec.delta_transform(x)
    np.testing.assert_allclose(result.numpy(), x.numpy() - 1.0, atol=1e-7)
    assert np.all(result.numpy() <= 0)


def test_delta_transform_straddle_scales_to_minus1_plus1() -> None:
    spec = get_payoff_spec(PayoffConfig(payoff_type="straddle"), _mc())
    import torch
    x = torch.tensor([0.0, 0.5, 1.0])
    result = spec.delta_transform(x)
    np.testing.assert_allclose(result.numpy(), np.array([-1.0, 0.0, 1.0]), atol=1e-7)


def test_dispatch_invalid_type_raises() -> None:
    import dataclasses
    bad_cfg = dataclasses.replace(PayoffConfig(payoff_type="call"), payoff_type="binary")  # type: ignore
    with pytest.raises((ValueError, Exception)):
        get_payoff_spec(bad_cfg, _mc())  # type: ignore[arg-type]
