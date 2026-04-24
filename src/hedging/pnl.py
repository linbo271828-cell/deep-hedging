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
    payoff_fn: Callable[[np.ndarray], np.ndarray] | None = None,
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
        Strike price (used only when payoff_fn is None — default call payoff).
    r:
        Risk-free rate (continuous compounding).
    sigma:
        Volatility (passed through to delta_fn for BS-based strategies).
    cost_rate:
        Proportional transaction cost per unit of notional traded (e.g. 0.0005 = 5 bps).
    initial_option_price:
        Premium received at t=0 for selling the option.
    payoff_fn:
        Terminal payoff callable: payoff_fn(s_T) -> np.ndarray, shape (n_paths,).
        If None, defaults to European call payoff: max(s_T - k, 0).

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

    payoff = payoff_fn(s_T) if payoff_fn is not None else np.maximum(s_T - k, 0.0)
    cash = cash - payoff

    return cash


def hedge_pnl_multi(
    paths: np.ndarray,
    hedge_option_prices: np.ndarray,
    stock_delta_fn: Callable[[np.ndarray, float], np.ndarray],
    hedge_delta_fn: Callable[[np.ndarray, float], np.ndarray],
    time_grid: np.ndarray,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    initial_option_price: float,
    cost_rate: float,
    r: float,
    k_h: float,
) -> np.ndarray:
    """Simulate P&L for a two-instrument delta-hedging strategy (stock + option).

    The hedger holds positions in both the underlying stock and a liquid hedge option.
    Both legs are rebalanced at each time step with proportional transaction costs.
    The hedge option expires at T with payoff max(S_T - k_h, 0); no liquidation cost
    is applied at expiry since the option settles naturally.

    Parameters
    ----------
    paths:
        Stock price paths, shape (n_paths, n_steps+1).
    hedge_option_prices:
        Hedge option prices along the same paths, shape (n_paths, n_steps+1).
        Computed externally via StockPlusOptionUniverse.
    stock_delta_fn:
        Callable(s_array, tau) -> stock hedge ratio, shape (n_paths,).
    hedge_delta_fn:
        Callable(s_array, tau) -> hedge-option position, shape (n_paths,).
    time_grid:
        Uniform time points [0, T], shape (n_steps+1,).
    payoff_fn:
        Terminal payoff of the main option: payoff_fn(s_T) -> np.ndarray.
    initial_option_price:
        Premium received at t=0 for selling the main option.
    cost_rate:
        Proportional transaction cost rate for both instruments.
    r:
        Risk-free rate (continuous compounding).
    k_h:
        Strike of the hedge option, used to compute terminal payoff.

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
    h0 = hedge_option_prices[:, 0]
    tau_0 = T

    ds = stock_delta_fn(s0, tau_0)
    dh = hedge_delta_fn(s0, tau_0)

    cost_s = proportional_cost(ds, s0, cost_rate)
    cost_h = proportional_cost(dh, h0, cost_rate)
    cash = initial_option_price - ds * s0 - cost_s - dh * h0 - cost_h
    cash = cash * np.exp(r * dt)

    for i in range(1, n_steps):
        s_i = paths[:, i]
        h_i = hedge_option_prices[:, i]
        tau_i = T - time_grid[i]

        ds_new = stock_delta_fn(s_i, tau_i)
        dh_new = hedge_delta_fn(s_i, tau_i)

        trade_s = ds_new - ds
        trade_h = dh_new - dh
        cost_s = proportional_cost(trade_s, s_i, cost_rate)
        cost_h = proportional_cost(trade_h, h_i, cost_rate)

        cash = cash - trade_s * s_i - cost_s - trade_h * h_i - cost_h
        ds = ds_new
        dh = dh_new
        cash = cash * np.exp(r * dt)

    s_T = paths[:, n_steps]
    liq_cost_s = proportional_cost(ds, s_T, cost_rate)
    cash = cash + ds * s_T - liq_cost_s

    hedge_payoff = np.maximum(s_T - k_h, 0.0)
    cash = cash + dh * hedge_payoff

    main_payoff = payoff_fn(s_T)
    cash = cash - main_payoff

    return cash
