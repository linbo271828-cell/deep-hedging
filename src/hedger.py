"""P&L accounting engine and classical Black-Scholes delta hedger.

This module owns the core simulation of a delta-hedging strategy for a short
European call position under proportional transaction costs.

Math reference: writeup/math_reference.md §3.

The hedger simulates selling one European call at t=0, collecting the premium,
and rebalancing a stock position at each time step to maintain delta-neutrality.
At maturity the stock position is liquidated and the call payoff is settled.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src import black_scholes as bs


def hedge_pnl(
    paths: np.ndarray,
    delta_fn: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray],
    time_grid: np.ndarray,
    k: float,
    r: float,
    sigma: float,
    cost_rate: float,
    initial_option_price: float,
) -> np.ndarray:
    """Simulate P&L for a delta-hedging strategy on a short call position.

    Models selling one European call at t=0 and dynamically delta-hedging.

    P&L accounting per path:
      t=0: receive option premium; buy delta_0 shares; pay proportional cost.
      t=i: rebalance to new delta; pay cost on |trade| * S_i; cash earns r*dt.
      t=T: liquidate stock position; pay call payoff; terminal cash = P&L.

    The cash account compounds continuously at rate r between rebalancing steps.

    Parameters
    ----------
    paths:
        Asset price paths, shape (n_paths, n_steps + 1).
    delta_fn:
        Callable(s, tau) -> hedge ratio, where s and tau are arrays of shape
        (n_paths,). Returns an array of shape (n_paths,).
    time_grid:
        Uniform time points [0, T], shape (n_steps + 1,).
    k:
        Strike price.
    r:
        Risk-free rate (continuous compounding).
    sigma:
        Volatility (passed to delta_fn for BS-based strategies).
    cost_rate:
        Proportional transaction cost per unit of notional traded,
        e.g. 0.0005 = 5 basis points.
    initial_option_price:
        Premium received at t=0 for selling the call.

    Returns
    -------
    np.ndarray
        Terminal P&L for each path, shape (n_paths,).
    """
    n_paths, n_steps_plus_1 = paths.shape
    n_steps = n_steps_plus_1 - 1
    T = time_grid[-1]

    dt = T / n_steps

    # ----- t = 0: open position -----
    s0 = paths[:, 0]  # shape (n_paths,)
    tau_0 = T - time_grid[0]  # = T
    delta = delta_fn(s0, tau_0)  # shape (n_paths,)

    # Cash: receive premium, buy delta shares, pay transaction cost on purchase.
    # cost = delta * s0 * cost_rate  (all deltas are non-negative for a call)
    cost = np.abs(delta) * s0 * cost_rate
    cash = initial_option_price - delta * s0 - cost  # shape (n_paths,)

    # Compound cash for one period before step 1.
    cash = cash * np.exp(r * dt)

    # ----- t = 1 .. n_steps - 1: rebalance -----
    for i in range(1, n_steps):
        s_i = paths[:, i]
        tau_i = T - time_grid[i]
        delta_new = delta_fn(s_i, tau_i)

        trade = delta_new - delta  # positive = buy, negative = sell
        cost = np.abs(trade) * s_i * cost_rate

        # Cash decreases by the cost of shares bought plus transaction costs.
        cash = cash - trade * s_i - cost

        delta = delta_new

        # Compound cash before next step.
        cash = cash * np.exp(r * dt)

    # ----- t = T: close position -----
    s_T = paths[:, n_steps]

    # Liquidate stock: receive delta * S_T, pay liquidation cost.
    liquidation_cost = np.abs(delta) * s_T * cost_rate
    cash = cash + delta * s_T - liquidation_cost

    # Settle call payoff (short position: we pay the holder).
    payoff = np.maximum(s_T - k, 0.0)
    cash = cash - payoff

    return cash


def classical_delta_pnl(
    paths: np.ndarray,
    time_grid: np.ndarray,
    k: float,
    r: float,
    sigma: float,
    cost_rate: float,
    initial_option_price: float,
) -> np.ndarray:
    """Delta-hedge a short call using Black-Scholes delta, return terminal P&L.

    Thin wrapper around hedge_pnl that binds delta_fn to bs.call_delta.

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
        Implied volatility used for computing BS delta.
    cost_rate:
        Proportional transaction cost per unit of notional traded.
    initial_option_price:
        Premium received at t=0 (should be bs.call_price(s0, k, r, sigma, T)).

    Returns
    -------
    np.ndarray
        Terminal P&L for each path, shape (n_paths,).
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
