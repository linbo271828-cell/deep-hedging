"""Transaction cost models.

v1 scope: proportional transaction costs only.
Future: fixed costs, impact models (out of scope for v1).

These are pure functions.  All cost logic for the P&L engine is routed through
this module so that cost models can be changed in one place.
"""

from __future__ import annotations

import numpy as np


def proportional_cost(
    trade: np.ndarray | float,
    s: np.ndarray | float,
    cost_rate: float,
) -> np.ndarray | float:
    """Proportional transaction cost: |trade| * S * cost_rate.

    Parameters
    ----------
    trade:
        Change in hedge ratio (positive = buy, negative = sell).
    s:
        Current asset price.
    cost_rate:
        Cost per unit of notional traded (e.g. 0.0005 = 5 bps).

    Returns
    -------
    Non-negative cost, same shape as trade/s.
    """
    return np.abs(trade) * np.asarray(s) * cost_rate
