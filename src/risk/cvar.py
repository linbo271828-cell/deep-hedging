"""Conditional Value-at-Risk (Expected Shortfall).

Pure function: numpy array in, float out.
Used as both training loss (via differentiable torch version in src/neural/train.py)
and evaluation metric in experiments.

Sign convention: higher return = MORE risk = WORSE outcome.
"""

from __future__ import annotations

import math

import numpy as np


def cvar(pnl: np.ndarray, alpha: float = 0.95) -> float:
    """CVaR (Expected Shortfall) at confidence level alpha.

    CVaR_alpha is the expected loss in the worst (1-alpha) fraction of outcomes.
    It is a coherent risk measure: convex, monotone, sub-additive,
    translation-invariant.

    Parameters
    ----------
    pnl:
        Terminal P&L per path, shape (n_paths,). Positive = gain, negative = loss.
    alpha:
        Confidence level, e.g. 0.95 means we examine the worst 5%.

    Returns
    -------
    float
        CVaR as a non-negative scalar. Larger = worse tail risk.

    Edge cases
    ----------
    - alpha=0.0 : mean loss (= -mean(pnl))
    - alpha=1.0 : maximum loss (= -min(pnl))

    Examples
    --------
    >>> import numpy as np
    >>> cvar(np.full(1000, -5.0))
    5.0
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    losses = -np.asarray(pnl, dtype=float)

    if alpha == 1.0:
        return float(losses.max())

    n = len(losses)
    n_tail = max(1, math.ceil((1.0 - alpha) * n))
    sorted_losses = np.sort(losses)[::-1]
    return float(sorted_losses[:n_tail].mean())
