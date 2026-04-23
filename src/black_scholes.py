"""Black-Scholes closed-form pricer and Greeks for European calls and puts.

The heart of Day 2 of the sprint.  Used as:
  - Sanity check for the Monte Carlo pricer in experiments/01.
  - Classical baseline hedger (delta) in experiments/02.
  - Ground truth that the neural hedger (no costs) must converge to.

Math reference: writeup/math_reference.md §2 (derivation of BS PDE and
formulas).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

_EPS = 1e-12


def _d1(
    s: np.ndarray | float,
    k: float,
    r: float,
    sigma: float,
    tau: np.ndarray | float,
) -> np.ndarray | float:
    """d1 of the Black-Scholes formula.

    d1 = (log(S/K) + (r + sigma^2 / 2) * tau) / (sigma * sqrt(tau))

    `tau` is time-to-maturity.  Guarded against tau=0 with _EPS so callers
    can evaluate right up to expiry without ZeroDivision.
    """
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


def call_price(
    s: np.ndarray | float,
    k: float,
    r: float,
    sigma: float,
    tau: np.ndarray | float,
) -> np.ndarray | float:
    """European call price under Black-Scholes.

    C = S * N(d1) - K * exp(-r * tau) * N(d2)
    """
    tau_safe = np.maximum(tau, _EPS)
    d1 = _d1(s, k, r, sigma, tau_safe)
    d2 = _d2(s, k, r, sigma, tau_safe)
    price = np.asarray(s) * norm.cdf(d1) - k * np.exp(-r * tau_safe) * norm.cdf(d2)
    # At expiry, price collapses to intrinsic value.
    intrinsic = np.maximum(np.asarray(s) - k, 0.0)
    return np.where(np.asarray(tau) <= 0, intrinsic, price)


def put_price(
    s: np.ndarray | float,
    k: float,
    r: float,
    sigma: float,
    tau: np.ndarray | float,
) -> np.ndarray | float:
    """European put via put-call parity: P = C - S + K * exp(-r * tau)."""
    return call_price(s, k, r, sigma, tau) - np.asarray(s) + k * np.exp(-r * np.maximum(tau, _EPS))


def call_delta(
    s: np.ndarray | float,
    k: float,
    r: float,
    sigma: float,
    tau: np.ndarray | float,
) -> np.ndarray | float:
    """dC/dS = N(d1).  At expiry, 1 if ITM else 0."""
    d1 = _d1(s, k, r, sigma, tau)
    delta = norm.cdf(d1)
    at_expiry = np.where(np.asarray(s) > k, 1.0, 0.0)
    return np.where(np.asarray(tau) <= 0, at_expiry, delta)


def call_gamma(
    s: np.ndarray | float,
    k: float,
    r: float,
    sigma: float,
    tau: np.ndarray | float,
) -> np.ndarray | float:
    """d^2 C / dS^2 = phi(d1) / (S * sigma * sqrt(tau))."""
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
    """dC/dt (time decay, per unit time).

    theta = -S * phi(d1) * sigma / (2 * sqrt(tau)) - r * K * exp(-r * tau) * N(d2)
    """
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
