"""Straddle payoff for Deep Hedging Lab.

A straddle is long a call and long a put at the same strike k.
Terminal payoff = |S_T − K|.

Classical benchmark:
    Net BS delta = call_delta + put_delta = 2·N(d₁) − 1.
    This is approximately 0 at-the-money, as expected by symmetry.
    Computed in src/payoffs/dispatch.py.
"""

from __future__ import annotations

import numpy as np


def straddle_payoff(
    s_T: np.ndarray | float, k: float
) -> np.ndarray | float:
    """Straddle terminal payoff: max(S_T−K,0) + max(K−S_T,0) = |S_T − K|."""
    s = np.asarray(s_T)
    return np.abs(s - k)
