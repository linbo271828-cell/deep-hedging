"""Straddle payoff — STUB, not yet implemented.

Lane C (payoffs and baselines) owns this file.
Implement after spreads are working (Phase B).

A straddle is long a call and long a put at the same strike k.

Target public interface:
    straddle_payoff(s_T, k) -> ndarray | float

Classical baseline: the call delta is N(d1) and put delta is N(d1)-1,
so the combined straddle delta = 2*N(d1) - 1 (≈ 0 at-the-money, as expected).
"""

from __future__ import annotations

import numpy as np


def straddle_payoff(
    s_T: np.ndarray | float, k: float
) -> np.ndarray | float:
    """Straddle terminal payoff: max(S_T-K,0) + max(K-S_T,0) = |S_T - K|.

    STUB — not yet implemented.
    """
    raise NotImplementedError(
        "straddle_payoff is not yet implemented. "
        "See writeup/implementation_plan.md Phase 2."
    )
