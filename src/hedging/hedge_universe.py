"""Hedge universe implementations — which instruments the hedger can trade.

v1 universes:
  StockOnlyUniverse       : single risky asset (the underlying stock).
  StockPlusOptionUniverse : stock + one liquid European call (the hedge option).

Hedge option design (stock_plus_option):
  - Instrument  : European call on the same underlying.
  - Strike      : HedgeUniverseConfig.hedge_option_strike (default 100 = ATM).
  - Maturity    : HedgeUniverseConfig.hedge_option_maturity (default T, co-terminous).
  - Pricing     : Black-Scholes call_price at each time step.
  - Costs       : same proportional cost_rate applied to |trade * option_price| notional.
  - Terminal    : hedge option expires at its intrinsic value max(S_T - k_h, 0);
                  no additional transaction cost at expiry (natural exercise).

Classical benchmark for stock_plus_option:
  Stock leg  : same payoff-specific BS delta as the stock_only benchmark.
  Hedge leg  : ZERO — the classical baseline holds no hedge-option position.
  Rationale  : There is no closed-form optimal policy for stock + option hedging
               in a GBM world; using zero is honest and makes the comparison
               deliberately asymmetric in the neural hedger's favour.
               This is disclosed in all metadata and captions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.payoffs.european import call_price

if TYPE_CHECKING:
    from src.experiments.configs import HedgeUniverseConfig, MarketConfig


class StockOnlyUniverse:
    """Single tradable instrument: the underlying stock."""

    @property
    def n_instruments(self) -> int:
        return 1

    def instrument_prices(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
    ) -> np.ndarray:
        """Return stock prices, shape (n_paths, n_steps+1, 1)."""
        return paths[:, :, np.newaxis]


class StockPlusOptionUniverse:
    """Two tradable instruments: underlying stock + one liquid European call.

    Parameters
    ----------
    market_config:
        Market parameters (r, sigma for option pricing).
    universe_config:
        hedge_option_strike and hedge_option_maturity.
    """

    def __init__(
        self,
        market_config: "MarketConfig",
        universe_config: "HedgeUniverseConfig",
    ) -> None:
        self._r = market_config.r
        self._sigma = market_config.sigma
        self._k_h = universe_config.hedge_option_strike
        self._T_h = universe_config.hedge_option_maturity

    @property
    def n_instruments(self) -> int:
        return 2

    @property
    def k_h(self) -> float:
        return self._k_h

    @property
    def T_h(self) -> float:
        return self._T_h

    def hedge_option_price(self, s: np.ndarray, tau: float) -> np.ndarray:
        """Black-Scholes price of the hedge option at time tau before expiry."""
        if tau <= 0.0:
            return np.maximum(np.asarray(s) - self._k_h, 0.0)
        return np.asarray(
            call_price(np.asarray(s), self._k_h, self._r, self._sigma, tau),
            dtype=float,
        )

    def hedge_option_prices_along_paths(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
    ) -> np.ndarray:
        """Price the hedge option at every (path, step) point.

        Returns shape (n_paths, n_steps+1).
        """
        n_paths, n_steps_plus_1 = paths.shape
        prices = np.empty((n_paths, n_steps_plus_1), dtype=float)
        for t_idx in range(n_steps_plus_1):
            tau = max(0.0, self._T_h - float(time_grid[t_idx]))
            prices[:, t_idx] = self.hedge_option_price(paths[:, t_idx], tau)
        return prices

    def instrument_prices(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
    ) -> np.ndarray:
        """Return both instrument prices, shape (n_paths, n_steps+1, 2).

        [:, :, 0] = stock prices
        [:, :, 1] = hedge option prices
        """
        hedge_prices = self.hedge_option_prices_along_paths(paths, time_grid)
        return np.stack([paths, hedge_prices], axis=-1)


def make_universe(
    market_config: "MarketConfig",
    universe_config: "HedgeUniverseConfig",
) -> "StockOnlyUniverse | StockPlusOptionUniverse":
    """Factory: build the correct hedge universe from config."""
    if universe_config.universe_type == "stock_only":
        return StockOnlyUniverse()
    if universe_config.universe_type == "stock_plus_option":
        return StockPlusOptionUniverse(market_config, universe_config)
    raise ValueError(f"Unknown universe_type: {universe_config.universe_type!r}")
