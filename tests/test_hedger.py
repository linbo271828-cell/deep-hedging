"""Correctness tests for the P&L accounting engine and classical delta hedger.

Numerical claims enforced here:
  1. With zero transaction costs under risk-neutral GBM, mean P&L is near zero
     (within 3 standard errors) — the hedger is self-financing in expectation.
  2. Transaction costs strictly reduce mean P&L.
  3. hedge_pnl returns the correct shape (n_paths,).
  4. classical_delta_pnl produces identical results to hedge_pnl when the
     same Black-Scholes delta function is passed explicitly.
  5. With a single rebalancing step and known parameters the P&L is manually
     traceable (regression / consistency check).
"""

from __future__ import annotations

import numpy as np
import pytest

from src import black_scholes as bs
from src.hedger import classical_delta_pnl, hedge_pnl
from src.market import GBM, time_grid


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def market_params() -> dict:
    return {
        "s0": 100.0,
        "r": 0.02,
        "sigma": 0.20,
        "k": 100.0,
        "T": 0.5,
    }


# ---------------------------------------------------------------------------
# Test 1: Zero-cost, risk-neutral GBM -> mean P&L ≈ 0
# ---------------------------------------------------------------------------

def test_pnl_zero_cost_near_zero_mean(market_params: dict) -> None:
    """Under risk-neutral GBM and zero costs, mean hedger P&L should be ~0.

    The Black-Scholes hedger is self-financing in the risk-neutral measure.
    With 10k paths and 252 rebalancing steps the discretisation bias is small
    enough that the mean should fall within 3 standard errors of zero.
    """
    s0 = market_params["s0"]
    r = market_params["r"]
    sigma = market_params["sigma"]
    k = market_params["k"]
    T = market_params["T"]
    n_steps = 252
    n_paths = 10_000
    seed = 7

    # Risk-neutral GBM: mu = r
    gbm = GBM(s0=s0, mu=r, sigma=sigma)
    paths = gbm.sample_paths(n_paths=n_paths, n_steps=n_steps, T=T, seed=seed)
    tg = time_grid(n_steps=n_steps, T=T)

    option_price = float(bs.call_price(s0, k, r, sigma, T))

    pnl = classical_delta_pnl(
        paths=paths,
        time_grid=tg,
        k=k,
        r=r,
        sigma=sigma,
        cost_rate=0.0,
        initial_option_price=option_price,
    )

    assert pnl.shape == (n_paths,), f"Expected shape ({n_paths},), got {pnl.shape}"

    mean_pnl = pnl.mean()
    se = pnl.std(ddof=1) / np.sqrt(n_paths)

    assert abs(mean_pnl) < 3 * se, (
        f"Mean P&L {mean_pnl:.6f} is more than 3 SE ({se:.6f}) from zero. "
        f"Hedger may not be self-financing."
    )


# ---------------------------------------------------------------------------
# Test 2: Transaction costs strictly reduce mean P&L
# ---------------------------------------------------------------------------

def test_costs_reduce_mean_pnl(market_params: dict) -> None:
    """A hedger with positive transaction costs must earn less than one without.

    With 5k paths and cost_rate=5bps, the difference in mean P&L should be
    clearly negative (more than 2 SEs below zero).
    """
    s0 = market_params["s0"]
    r = market_params["r"]
    sigma = market_params["sigma"]
    k = market_params["k"]
    T = market_params["T"]
    n_steps = 50
    n_paths = 5_000
    seed = 42

    gbm = GBM(s0=s0, mu=r, sigma=sigma)
    paths = gbm.sample_paths(n_paths=n_paths, n_steps=n_steps, T=T, seed=seed)
    tg = time_grid(n_steps=n_steps, T=T)

    option_price = float(bs.call_price(s0, k, r, sigma, T))

    pnl_no_cost = classical_delta_pnl(
        paths=paths,
        time_grid=tg,
        k=k,
        r=r,
        sigma=sigma,
        cost_rate=0.0,
        initial_option_price=option_price,
    )

    pnl_with_cost = classical_delta_pnl(
        paths=paths,
        time_grid=tg,
        k=k,
        r=r,
        sigma=sigma,
        cost_rate=0.0005,  # 5 bps
        initial_option_price=option_price,
    )

    diff = pnl_with_cost - pnl_no_cost  # should be strictly negative
    mean_diff = diff.mean()
    se_diff = diff.std(ddof=1) / np.sqrt(n_paths)

    assert mean_diff < 0, (
        f"Expected costs to reduce P&L but mean diff = {mean_diff:.6f}"
    )
    assert mean_diff < -2 * se_diff, (
        f"Cost reduction ({mean_diff:.6f}) is not statistically significant "
        f"(SE = {se_diff:.6f}). Expected at least 2 SE separation."
    )


# ---------------------------------------------------------------------------
# Test 3: Output shape
# ---------------------------------------------------------------------------

def test_pnl_shape(market_params: dict) -> None:
    """hedge_pnl must return an array of shape (n_paths,)."""
    s0 = market_params["s0"]
    r = market_params["r"]
    sigma = market_params["sigma"]
    k = market_params["k"]
    T = market_params["T"]
    n_paths = 8
    n_steps = 5

    gbm = GBM(s0=s0, mu=r, sigma=sigma)
    paths = gbm.sample_paths(n_paths=n_paths, n_steps=n_steps, T=T, seed=0)
    tg = time_grid(n_steps=n_steps, T=T)

    option_price = float(bs.call_price(s0, k, r, sigma, T))

    def delta_fn(s: np.ndarray, tau: float | np.ndarray) -> np.ndarray:
        return np.asarray(bs.call_delta(s, k, r, sigma, tau))

    pnl = hedge_pnl(
        paths=paths,
        delta_fn=delta_fn,
        time_grid=tg,
        k=k,
        r=r,
        sigma=sigma,
        cost_rate=0.0,
        initial_option_price=option_price,
    )

    assert pnl.shape == (n_paths,), f"Expected ({n_paths},), got {pnl.shape}"
    assert np.isfinite(pnl).all(), "P&L contains NaN or Inf"


# ---------------------------------------------------------------------------
# Test 4: classical_delta_pnl matches hedge_pnl with explicit BS delta_fn
# ---------------------------------------------------------------------------

def test_classical_delta_pnl_matches_hedge_pnl(market_params: dict) -> None:
    """classical_delta_pnl must produce identical results to hedge_pnl when
    called with the equivalent Black-Scholes delta function."""
    s0 = market_params["s0"]
    r = market_params["r"]
    sigma = market_params["sigma"]
    k = market_params["k"]
    T = market_params["T"]
    n_paths = 200
    n_steps = 20

    gbm = GBM(s0=s0, mu=r, sigma=sigma)
    paths = gbm.sample_paths(n_paths=n_paths, n_steps=n_steps, T=T, seed=13)
    tg = time_grid(n_steps=n_steps, T=T)

    option_price = float(bs.call_price(s0, k, r, sigma, T))
    cost_rate = 0.001

    # Via wrapper
    pnl_wrapper = classical_delta_pnl(
        paths=paths,
        time_grid=tg,
        k=k,
        r=r,
        sigma=sigma,
        cost_rate=cost_rate,
        initial_option_price=option_price,
    )

    # Via explicit delta_fn
    def delta_fn(s: np.ndarray, tau: np.ndarray | float) -> np.ndarray:
        return np.asarray(bs.call_delta(s, k, r, sigma, tau))

    pnl_explicit = hedge_pnl(
        paths=paths,
        delta_fn=delta_fn,
        time_grid=tg,
        k=k,
        r=r,
        sigma=sigma,
        cost_rate=cost_rate,
        initial_option_price=option_price,
    )

    np.testing.assert_array_equal(
        pnl_wrapper,
        pnl_explicit,
        err_msg="classical_delta_pnl and hedge_pnl with BS delta produce different results",
    )


# ---------------------------------------------------------------------------
# Test 5: Single-step manual trace (regression / consistency)
# ---------------------------------------------------------------------------

def test_single_step_manual_trace() -> None:
    """With n_steps=1, manually trace through the P&L formula and check.

    Setup:
      - s0 = 100, S_T = 110 (known terminal price, constant path)
      - k = 100, r = 0, sigma = 0.2, T = 0.5
      - cost_rate = 0 for clarity
      - delta_fn always returns a fixed value delta_val

    Expected P&L:
      cash_0  = option_price - delta_val * 100        (bought delta_val shares)
      cash_0 *= exp(r * T) = cash_0 (since r=0)
      At T: cash_0 + delta_val * 110 - max(110-100, 0)
           = option_price - delta_val*100 + delta_val*110 - 10
           = option_price + delta_val*10 - 10
    """
    s0 = 100.0
    s_T = 110.0
    k = 100.0
    r = 0.0
    sigma = 0.2
    T = 0.5
    delta_val = 0.6
    cost_rate = 0.0

    option_price = float(bs.call_price(s0, k, r, sigma, T))

    # Construct a constant path: [s0, s_T]
    paths = np.array([[s0, s_T]])  # shape (1, 2)
    tg = time_grid(n_steps=1, T=T)

    def delta_fn(s: np.ndarray, tau: np.ndarray | float) -> np.ndarray:
        return np.full_like(np.asarray(s, dtype=float), delta_val)

    pnl = hedge_pnl(
        paths=paths,
        delta_fn=delta_fn,
        time_grid=tg,
        k=k,
        r=r,
        sigma=sigma,
        cost_rate=cost_rate,
        initial_option_price=option_price,
    )

    expected_pnl = option_price + delta_val * (s_T - s0) - max(s_T - k, 0.0)
    np.testing.assert_allclose(
        pnl,
        np.array([expected_pnl]),
        atol=1e-10,
        err_msg=(
            f"Single-step P&L mismatch: got {pnl[0]:.8f}, "
            f"expected {expected_pnl:.8f}"
        ),
    )
