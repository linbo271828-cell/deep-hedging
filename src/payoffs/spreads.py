"""Vertical spread payoffs for Deep Hedging Lab.

A bull call spread is long a call at k_lo and short a call at k_hi (k_lo < k_hi).
A bear put spread is long a put at k_hi and short a put at k_lo (k_lo < k_hi).

Classical benchmark (no single BS delta):
    The net BS delta is call_delta(k_lo) − call_delta(k_hi) for the call spread,
    or put_delta(k_hi) − put_delta(k_lo) for the put spread.
    These are computed in src/payoffs/dispatch.py, not here.
"""

from __future__ import annotations

import numpy as np


def call_spread_payoff(
    s_T: np.ndarray | float, k_lo: float, k_hi: float
) -> np.ndarray | float:
    """Bull call spread terminal payoff: max(S_T−k_lo,0) − max(S_T−k_hi,0)."""
    s = np.asarray(s_T)
    return np.maximum(s - k_lo, 0.0) - np.maximum(s - k_hi, 0.0)


def put_spread_payoff(
    s_T: np.ndarray | float, k_lo: float, k_hi: float
) -> np.ndarray | float:
    """Bear put spread terminal payoff: max(k_hi−S_T,0) − max(k_lo−S_T,0)."""
    s = np.asarray(s_T)
    return np.maximum(k_hi - s, 0.0) - np.maximum(k_lo - s, 0.0)
