"""Vertical spread payoffs — STUB, not yet implemented.

Lane C (payoffs and baselines) owns this file.
Implement after the European-call baseline migration is complete (Phase B).

A vertical spread is long one call at strike k_lo and short one call at k_hi
(k_lo < k_hi), or equivalently for puts.

Target public interface:
    call_spread_payoff(s_T, k_lo, k_hi) -> ndarray | float
    put_spread_payoff(s_T, k_lo, k_hi) -> ndarray | float

No closed-form BS delta exists for a spread in general (it is the difference of
two BS deltas), so the classical baseline for spreads is the *component-wise*
BS delta hedge.  See src/hedging/baseline.py for details.
"""

from __future__ import annotations

import numpy as np


def call_spread_payoff(
    s_T: np.ndarray | float, k_lo: float, k_hi: float
) -> np.ndarray | float:
    """Bull call spread terminal payoff: max(S_T-k_lo,0) - max(S_T-k_hi,0).

    STUB — not yet implemented.
    """
    raise NotImplementedError(
        "call_spread_payoff is not yet implemented. "
        "See writeup/implementation_plan.md Phase 2."
    )


def put_spread_payoff(
    s_T: np.ndarray | float, k_lo: float, k_hi: float
) -> np.ndarray | float:
    """Bear put spread terminal payoff: max(k_hi-S_T,0) - max(k_lo-S_T,0).

    STUB — not yet implemented.
    """
    raise NotImplementedError(
        "put_spread_payoff is not yet implemented. "
        "See writeup/implementation_plan.md Phase 2."
    )
