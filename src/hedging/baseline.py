"""Classical Black-Scholes baseline hedgers.

Thin wrappers around hedge_pnl that bind the delta function to closed-form
Black-Scholes Greeks from src/payoffs/european.py.

Rule: if no exact classical benchmark exists for a payoff/model combination,
this module must say so explicitly (raise NotImplementedError or return None
with a docstring explanation).  Never silently apply an incorrect baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.hedging.pnl import hedge_pnl, hedge_pnl_multi
from src.payoffs import european as bs

if TYPE_CHECKING:
    from src.payoffs.dispatch import PayoffSpec


def classical_delta_pnl_generalized(
    paths: np.ndarray,
    time_grid: np.ndarray,
    payoff_spec: "PayoffSpec",
    cost_rate: float,
) -> np.ndarray:
    """Delta-hedge a short option position using the payoff spec's classical delta.

    Works for any payoff type registered in src/payoffs/dispatch.py.
    The classical_delta_fn and payoff_fn are taken from the PayoffSpec so that
    the benchmark is always consistent with the training payoff.

    Parameters
    ----------
    paths:
        Asset price paths, shape (n_paths, n_steps + 1).
    time_grid:
        Uniform time points [0, T], shape (n_steps + 1,).
    payoff_spec:
        PayoffSpec from get_payoff_spec(payoff_config, market_config).
    cost_rate:
        Proportional transaction cost per unit of notional traded.

    Returns
    -------
    np.ndarray
        Terminal P&L per path, shape (n_paths,).
    """
    return hedge_pnl(
        paths=paths,
        delta_fn=payoff_spec.classical_delta_fn,
        time_grid=time_grid,
        k=0.0,
        r=0.0,
        sigma=0.0,
        cost_rate=cost_rate,
        initial_option_price=payoff_spec.initial_option_price,
        payoff_fn=payoff_spec.payoff_fn,
    )


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


def classical_delta_pnl_stock_option(
    paths: np.ndarray,
    hedge_option_prices: np.ndarray,
    time_grid: np.ndarray,
    payoff_spec: "PayoffSpec",
    cost_rate: float,
    k_h: float,
) -> np.ndarray:
    """Classical benchmark for the stock + option hedge universe.

    Stock leg : payoff-specific BS delta (same as stock_only benchmark).
    Hedge leg : ZERO — the classical baseline does not use the hedge option.

    This is an explicitly asymmetric comparison: the neural hedger has access
    to both instruments while the classical baseline uses only the stock.
    The asymmetry is intentional and documented; it highlights the additional
    value the neural hedger can extract from the hedge option instrument.

    Parameters
    ----------
    paths:
        Stock price paths, shape (n_paths, n_steps+1).
    hedge_option_prices:
        Precomputed hedge option prices, shape (n_paths, n_steps+1).
    time_grid:
        Uniform time points [0, T], shape (n_steps+1,).
    payoff_spec:
        PayoffSpec for the main option (provides stock delta and payoff).
    cost_rate:
        Proportional transaction cost rate.
    k_h:
        Strike of the hedge option (used for terminal hedge-option payoff = 0).

    Returns
    -------
    np.ndarray
        Terminal P&L per path, shape (n_paths,).
    """
    n_paths = paths.shape[0]

    def zero_hedge_fn(s: np.ndarray, tau: float) -> np.ndarray:
        return np.zeros(len(np.atleast_1d(s)), dtype=float)

    return hedge_pnl_multi(
        paths=paths,
        hedge_option_prices=hedge_option_prices,
        stock_delta_fn=payoff_spec.classical_delta_fn,
        hedge_delta_fn=zero_hedge_fn,
        time_grid=time_grid,
        payoff_fn=payoff_spec.payoff_fn,
        initial_option_price=payoff_spec.initial_option_price,
        cost_rate=cost_rate,
        r=0.0,
        k_h=k_h,
    )
