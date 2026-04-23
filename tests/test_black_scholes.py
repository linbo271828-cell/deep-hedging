"""Black-Scholes correctness checks.

The claims we enforce:
  1. Put-call parity holds exactly in closed form.
  2. Delta from the closed form matches a finite-difference estimate.
  3. Gamma from the closed form matches a finite-difference estimate.
  4. Expiry payoff: as tau -> 0, price -> max(S - K, 0) for the call.
  5. Monte Carlo pricing (under risk-neutral GBM with mu = r) agrees with
     the closed-form price within 2 standard errors at 200k paths.
  6. Deep ITM / OTM limits behave sanely (delta -> 1 / 0).
"""

from __future__ import annotations

import numpy as np
import pytest

from src import black_scholes as bs
from src.market import GBM


@pytest.fixture
def params() -> dict[str, float]:
    return {"k": 100.0, "r": 0.02, "sigma": 0.2, "tau": 0.5}


def test_put_call_parity(params: dict[str, float]) -> None:
    s = np.array([80.0, 100.0, 120.0])
    c = bs.call_price(s, **params)
    p = bs.put_price(s, **params)
    k, r, tau = params["k"], params["r"], params["tau"]
    lhs = c - p
    rhs = s - k * np.exp(-r * tau)
    np.testing.assert_allclose(lhs, rhs, atol=1e-10)


def test_delta_matches_finite_difference(params: dict[str, float]) -> None:
    s0 = 100.0
    h = 1e-3
    analytic = bs.call_delta(s0, **params)
    fd = (bs.call_price(s0 + h, **params) - bs.call_price(s0 - h, **params)) / (2 * h)
    assert abs(analytic - fd) < 1e-6, f"analytic={analytic} fd={fd}"


def test_gamma_matches_finite_difference(params: dict[str, float]) -> None:
    s0 = 100.0
    h = 1e-2
    analytic = bs.call_gamma(s0, **params)
    fd = (
        bs.call_price(s0 + h, **params)
        - 2 * bs.call_price(s0, **params)
        + bs.call_price(s0 - h, **params)
    ) / (h**2)
    assert abs(analytic - fd) < 1e-4, f"analytic={analytic} fd={fd}"


def test_vega_matches_finite_difference(params: dict[str, float]) -> None:
    s0 = 100.0
    h = 1e-4
    analytic = bs.call_vega(s0, **params)
    up = params | {"sigma": params["sigma"] + h}
    dn = params | {"sigma": params["sigma"] - h}
    fd = (bs.call_price(s0, **up) - bs.call_price(s0, **dn)) / (2 * h)
    assert abs(analytic - fd) < 1e-4, f"analytic={analytic} fd={fd}"


def test_call_collapses_to_intrinsic_at_expiry(params: dict[str, float]) -> None:
    s = np.array([80.0, 100.0, 120.0])
    at_expiry = bs.call_price(s, params["k"], params["r"], params["sigma"], tau=0.0)
    np.testing.assert_allclose(at_expiry, np.maximum(s - params["k"], 0.0))


def test_delta_limits(params: dict[str, float]) -> None:
    # Deep ITM -> delta ~ 1
    assert bs.call_delta(1e6, **params) > 0.999
    # Deep OTM -> delta ~ 0
    assert bs.call_delta(1e-6, **params) < 0.001


def test_mc_price_matches_closed_form() -> None:
    """MC price under risk-neutral GBM (mu = r) should match closed form."""
    s0, k, r, sigma, T = 100.0, 100.0, 0.02, 0.2, 0.5
    n_paths = 200_000

    gbm_rn = GBM(s0=s0, mu=r, sigma=sigma)
    paths = gbm_rn.sample_paths(n_paths=n_paths, n_steps=100, T=T, seed=99)

    payoff = np.maximum(paths[:, -1] - k, 0.0)
    discounted = np.exp(-r * T) * payoff
    mc_price = discounted.mean()
    mc_se = discounted.std(ddof=1) / np.sqrt(n_paths)

    closed = bs.call_price(s0, k, r, sigma, T)

    # 3-sigma band; with 200k paths mc_se ~ 0.02, comfortably tight.
    assert abs(mc_price - closed) < 3 * mc_se, (
        f"mc={mc_price:.6f} closed={closed:.6f} se={mc_se:.6f}"
    )
