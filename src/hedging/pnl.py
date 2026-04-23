"""Canonical P&L engine for delta-hedging a short option position.

This is the highest-risk shared file in the hedging core.
Both classical and neural paths route through hedge_pnl.
Any change to this function's signature or accounting logic must be
coordinated with Lane B (neural) and Lane C (payoffs/baselines).

Accounting model:
  t=0   : receive option premium; buy delta_0 shares; pay proportional cost.
  t=i   : observe new delta signal; compute trade = delta_new - delta_old;
          pay |trade| * S_i * cost_rate; cash earns continuous interest exp(r*dt).
  t=T   : liquidate stock; settle option payoff; cash = terminal P&L.

Transaction cost logic lives ONLY in src/hedging/transaction_costs.py and here.
Do not compute costs elsewhere.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.hedging.transaction_costs import proportional_cost


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

    Parameters
    ----------
    paths:
        Asset price paths, shape (n_paths, n_steps + 1).
    delta_fn:
        Callable(s, tau) -> hedge ratio, shape (n_paths,).
    time_grid:
        Uniform time points [0, T], shape (n_steps + 1,).
    k:
        Strike price.
    r:
        Risk-free rate (continuous compounding).
    sigma:
        Volatility (passed through to delta_fn for BS-based strategies).
    cost_rate:
        Proportional transaction cost per unit of notional traded (e.g. 0.0005 = 5 bps).
    initial_option_price:
        Premium received at t=0 for selling the call.

    Returns
    -------
    np.ndarray
        Terminal P&L per path, shape (n_paths,).
    """
    n_paths, n_steps_plus_1 = paths.shape
    n_steps = n_steps_plus_1 - 1
    T = time_grid[-1]
    dt = T / n_steps

    s0 = paths[:, 0]
    tau_0 = T - time_grid[0]
    delta = delta_fn(s0, tau_0)

    cost = proportional_cost(delta, s0, cost_rate)
    cash = initial_option_price - delta * s0 - cost
    cash = cash * np.exp(r * dt)

    for i in range(1, n_steps):
        s_i = paths[:, i]
        tau_i = T - time_grid[i]
        delta_new = delta_fn(s_i, tau_i)

        trade = delta_new - delta
        cost = proportional_cost(trade, s_i, cost_rate)
        cash = cash - trade * s_i - cost
        delta = delta_new
        cash = cash * np.exp(r * dt)

    s_T = paths[:, n_steps]
    liquidation_cost = proportional_cost(delta, s_T, cost_rate)
    cash = cash + delta * s_T - liquidation_cost

    payoff = np.maximum(s_T - k, 0.0)
    cash = cash - payoff

    return cash
