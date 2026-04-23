"""Hedge universe definitions — STUB, not yet implemented.

Lane B (hedging core) owns this file.
Implement after the stock-only baseline migration is complete (Phase D).

A hedge universe is the set of tradable instruments available to the hedger.

v1 universes:
  1. Stock-only: one risky asset (the underlying).
  2. Stock + one liquid option: the underlying plus a liquid vanilla option
     that can be traded to reduce higher-order Greeks (e.g. vega).

The hedge universe interface allows the P&L engine and neural hedger to be
parameterized by the set of instruments, enabling fair comparison experiments.

Target public interface:
    HedgeUniverse (Protocol or abstract base):
        n_instruments: int
        instrument_prices(paths, vol_paths, time_grid) -> ndarray (n_paths, n_steps+1, n_instruments)

    StockOnlyUniverse: trivial 1-instrument universe
    StockPlusOptionUniverse: 2-instrument universe (stock + liquid option)
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class HedgeUniverse(Protocol):
    """Protocol for any set of tradable hedge instruments.

    Implementations must be able to price all instruments in the universe
    at each time step given the current market state.
    """

    @property
    def n_instruments(self) -> int:
        """Number of tradable instruments in this universe."""
        ...

    def instrument_prices(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
    ) -> np.ndarray:
        """Return instrument prices at each time step.

        STUB — not yet implemented.

        Returns shape (n_paths, n_steps + 1, n_instruments).
        """
        ...


class StockOnlyUniverse:
    """Single tradable instrument: the underlying stock.

    STUB — not yet implemented.
    """

    @property
    def n_instruments(self) -> int:
        return 1

    def instrument_prices(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError(
            "StockOnlyUniverse.instrument_prices is not yet implemented. "
            "See writeup/implementation_plan.md Phase 1."
        )


class StockPlusOptionUniverse:
    """Two tradable instruments: stock + one liquid vanilla option.

    STUB — not yet implemented. Implement in Phase D.
    """

    @property
    def n_instruments(self) -> int:
        return 2

    def instrument_prices(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError(
            "StockPlusOptionUniverse.instrument_prices is not yet implemented. "
            "See writeup/implementation_plan.md Phase 4."
        )
