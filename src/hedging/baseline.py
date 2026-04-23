"""Classical Black-Scholes baseline hedgers.

Thin wrappers around hedge_pnl that bind the delta function to closed-form
Black-Scholes Greeks from src/payoffs/european.py.

Rule: if no exact classical benchmark exists for a payoff/model combination,
this module must say so explicitly (raise NotImplementedError or return None
with a docstring explanation).  Never silently apply an incorrect baseline.
"""

from __future__ import annotations

import numpy as np

from src.hedging.pnl import hedge_pnl
from src.payoffs import european as bs


def classical_delta_pnl(
    paths: np.ndarray,
    time_grid: np.ndarray,
    k: float,
    r: float,
    sigma: float,
    cost_rate: float,
    initial_option_price: float,
) -> np.ndarray:
    """Delta-hedge a short European call using Black-Scholes delta.

    Parameters
    ----------
    paths:
        Asset price paths, shape (n_paths, n_steps + 1).
    time_grid:
        Uniform time points [0, T], shape (n_steps + 1,).
    k:
        Strike price.
    r:
        Risk-free rate (continuous compounding).
    sigma:
        Implied volatility used to compute BS call delta.
    cost_rate:
        Proportional transaction cost per unit of notional traded.
    initial_option_price:
        Premium received at t=0 (should be bs.call_price(s0, k, r, sigma, T)).

    Returns
    -------
    np.ndarray
        Terminal P&L per path, shape (n_paths,).
    """

    def delta_fn(s: np.ndarray | float, tau: np.ndarray | float) -> np.ndarray:
        return np.asarray(bs.call_delta(s, k, r, sigma, tau))

    return hedge_pnl(
        paths=paths,
        delta_fn=delta_fn,
        time_grid=time_grid,
        k=k,
        r=r,
        sigma=sigma,
        cost_rate=cost_rate,
        initial_option_price=initial_option_price,
    )
