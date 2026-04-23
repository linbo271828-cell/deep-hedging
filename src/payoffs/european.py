"""European option pricing under Black-Scholes and terminal payoff definitions.

This module owns:
1. Terminal payoff functions (model-independent, pure numpy).
2. Black-Scholes closed-form pricing for European calls and puts.
3. All first-order Greeks used by the classical baseline hedger.

These are pure functions — no state, no simulation, no side effects.
They handle tau=0 cleanly (intrinsic value, not NaN).

Note on Black-Scholes scope:
    BS formulas are specific to the GBM/lognormal world.  They are placed
    here (alongside payoffs) because:
    - They are payoff-specific pricing functions, not general model machinery.
    - Under Heston the relevant analytical formula is different (Heston char-fn).
    - The classical baseline hedger (src/hedging/baseline.py) imports delta
      from here for the GBM+European-call case only.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Terminal payoff functions (model-independent)
# ---------------------------------------------------------------------------


def call_payoff(s_T: np.ndarray | float, k: float) -> np.ndarray | float:
    """European call terminal payoff: max(S_T - K, 0)."""
    return np.maximum(np.asarray(s_T) - k, 0.0)


def put_payoff(s_T: np.ndarray | float, k: float) -> np.ndarray | float:
    """European put terminal payoff: max(K - S_T, 0)."""
    return np.maximum(k - np.asarray(s_T), 0.0)


# ---------------------------------------------------------------------------
# Black-Scholes internals
# ---------------------------------------------------------------------------


def _d1(
    s: np.ndarray | float,
    k: float,
    r: float,
    sigma: float,
    tau: np.ndarray | float,
) -> np.ndarray | float:
    tau = np.maximum(tau, _EPS)
    return (np.log(np.asarray(s) / k) + (r + 0.5 * sigma**2) * tau) / (
        sigma * np.sqrt(tau)
    )


def _d2(
    s: np.ndarray | float,
    k: float,
    r: float,
    sigma: float,
    tau: np.ndarray | float,
) -> np.ndarray | float:
    tau = np.maximum(tau, _EPS)
    return _d1(s, k, r, sigma, tau) - sigma * np.sqrt(tau)


# ---------------------------------------------------------------------------
# Black-Scholes prices
# ---------------------------------------------------------------------------


def call_price(
    s: np.ndarray | float,
    k: float,
    r: float,
    sigma: float,
    tau: np.ndarray | float,
) -> np.ndarray | float:
    """European call price: C = S*N(d1) - K*exp(-r*tau)*N(d2)."""
    tau_safe = np.maximum(tau, _EPS)
    d1 = _d1(s, k, r, sigma, tau_safe)
    d2 = _d2(s, k, r, sigma, tau_safe)
    price = np.asarray(s) * norm.cdf(d1) - k * np.exp(-r * tau_safe) * norm.cdf(d2)
    intrinsic = np.maximum(np.asarray(s) - k, 0.0)
    return np.where(np.asarray(tau) <= 0, intrinsic, price)


def put_price(
    s: np.ndarray | float,
    k: float,
    r: float,
    sigma: float,
    tau: np.ndarray | float,
) -> np.ndarray | float:
    """European put price via put-call parity: P = C - S + K*exp(-r*tau)."""
    return (
        call_price(s, k, r, sigma, tau)
        - np.asarray(s)
        + k * np.exp(-r * np.maximum(tau, _EPS))
    )


# ---------------------------------------------------------------------------
# Black-Scholes Greeks
# ---------------------------------------------------------------------------


def call_delta(
    s: np.ndarray | float,
    k: float,
    r: float,
    sigma: float,
    tau: np.ndarray | float,
) -> np.ndarray | float:
    """dC/dS = N(d1).  At expiry: 1 if ITM else 0."""
    d1 = _d1(s, k, r, sigma, tau)
    delta = norm.cdf(d1)
    at_expiry = np.where(np.asarray(s) > k, 1.0, 0.0)
    return np.where(np.asarray(tau) <= 0, at_expiry, delta)


def put_delta(
    s: np.ndarray | float,
    k: float,
    r: float,
    sigma: float,
    tau: np.ndarray | float,
) -> np.ndarray | float:
    """dP/dS = N(d1) - 1.  At expiry: -1 if ITM else 0."""
    return call_delta(s, k, r, sigma, tau) - 1.0


def call_gamma(
    s: np.ndarray | float,
    k: float,
    r: float,
    sigma: float,
    tau: np.ndarray | float,
) -> np.ndarray | float:
    """d^2C/dS^2 = phi(d1) / (S * sigma * sqrt(tau))."""
    tau_safe = np.maximum(tau, _EPS)
    d1 = _d1(s, k, r, sigma, tau_safe)
    gamma = norm.pdf(d1) / (np.asarray(s) * sigma * np.sqrt(tau_safe))
    return np.where(np.asarray(tau) <= 0, 0.0, gamma)


def call_vega(
    s: np.ndarray | float,
    k: float,
    r: float,
    sigma: float,
    tau: np.ndarray | float,
) -> np.ndarray | float:
    """dC/d(sigma) = S * phi(d1) * sqrt(tau)."""
    tau_safe = np.maximum(tau, _EPS)
    d1 = _d1(s, k, r, sigma, tau_safe)
    vega = np.asarray(s) * norm.pdf(d1) * np.sqrt(tau_safe)
    return np.where(np.asarray(tau) <= 0, 0.0, vega)


def call_theta(
    s: np.ndarray | float,
    k: float,
    r: float,
    sigma: float,
    tau: np.ndarray | float,
) -> np.ndarray | float:
    """dC/dt = -S*phi(d1)*sigma/(2*sqrt(tau)) - r*K*exp(-r*tau)*N(d2)."""
    tau_safe = np.maximum(tau, _EPS)
    d1 = _d1(s, k, r, sigma, tau_safe)
    d2 = _d2(s, k, r, sigma, tau_safe)
    theta = (
        -np.asarray(s) * norm.pdf(d1) * sigma / (2.0 * np.sqrt(tau_safe))
        - r * k * np.exp(-r * tau_safe) * norm.cdf(d2)
    )
    return np.where(np.asarray(tau) <= 0, 0.0, theta)


def call_rho(
    s: np.ndarray | float,
    k: float,
    r: float,
    sigma: float,
    tau: np.ndarray | float,
) -> np.ndarray | float:
    """dC/dr = K * tau * exp(-r * tau) * N(d2)."""
    tau_safe = np.maximum(tau, _EPS)
    d2 = _d2(s, k, r, sigma, tau_safe)
    rho = k * tau_safe * np.exp(-r * tau_safe) * norm.cdf(d2)
    return np.where(np.asarray(tau) <= 0, 0.0, rho)
