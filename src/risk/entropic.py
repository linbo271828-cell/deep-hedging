"""Entropic risk measure (exponential utility certainty equivalent).

Pure function: numpy array in, float out.
Numerically stable via log-sum-exp.

Sign convention: higher return = MORE risk = WORSE outcome.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp


def entropic_risk(pnl: np.ndarray, lambda_: float = 1.0) -> float:
    """Entropic risk measure at risk-aversion coefficient lambda_.

    rho(X) = (1/lambda_) * log( E[exp(-lambda_ * X)] )

    Properties: convex, monotone, translation-invariant.
    For X ~ N(mu, sigma^2): rho = -mu + lambda_*sigma^2/2.

    As lambda_ -> 0: converges to -E[X] (mean loss).
    As lambda_ -> inf: converges to ess sup(-X) (worst case).

    Parameters
    ----------
    pnl:
        Terminal P&L per path, shape (n_paths,).
    lambda_:
        Risk-aversion coefficient. Must be > 0.

    Returns
    -------
    float
        Entropic risk. Larger = more tail risk.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> abs(entropic_risk(rng.standard_normal(200_000), lambda_=1.0) - 0.5) < 0.01
    True
    """
    if lambda_ <= 0.0:
        raise ValueError(f"lambda_ must be positive, got {lambda_}")

    pnl_arr = np.asarray(pnl, dtype=float)
    n = len(pnl_arr)
    v = -lambda_ * pnl_arr
    log_mean_exp = logsumexp(v) - math.log(n)
    return float(log_mean_exp / lambda_)
