"""Pure risk-measure functions for terminal P&L distributions.

Used both as training-loss functions (inside train.py) and as evaluation metrics
in experiments.  All functions are pure: no model dependencies, no side effects,
no global state.

Sign convention throughout:  higher return value = MORE risk = WORSE outcome.
This matches the convention in Buehler et al. (2019): the network minimises the
risk of the *hedged* P&L, so the loss function must increase when outcomes worsen.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp


def cvar(pnl: np.ndarray, alpha: float = 0.95) -> float:
    """Conditional Value-at-Risk (Expected Shortfall) at confidence level alpha.

    CVaR_alpha is the expected loss in the worst (1 - alpha) fraction of outcomes.
    It is always >= VaR_alpha (the alpha-quantile of losses) and is a coherent
    risk measure (sub-additive, convex, monotone, translation-invariant).

    Parameters
    ----------
    pnl : np.ndarray
        Terminal P&L for each simulated path, shape (n_paths,).
        Positive values are gains; negative values are losses.
    alpha : float
        Confidence level, typically 0.95 or 0.99.  Must be in [0, 1].
        alpha=0.95 means we look at the worst 5% of outcomes.

    Returns
    -------
    float
        CVaR as a non-negative scalar.  Larger values indicate worse tail risk.
        CVaR(pnl, alpha=0.95) is the mean loss in the worst 5% of P&L outcomes.

    Edge cases
    ----------
    - alpha=0.0 : returns the mean loss (= -mean(pnl)).
    - alpha=1.0 : returns the maximum loss (= -min(pnl)).

    Examples
    --------
    >>> import numpy as np
    >>> pnl = np.full(1000, -5.0)   # always lose 5
    >>> cvar(pnl)
    5.0
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    losses = -np.asarray(pnl, dtype=float)

    if alpha == 1.0:
        return float(losses.max())

    n = len(losses)
    # Number of worst outcomes to average over.
    n_tail = max(1, math.ceil((1.0 - alpha) * n))
    sorted_losses = np.sort(losses)[::-1]  # descending: worst first
    return float(sorted_losses[:n_tail].mean())


def entropic_risk(pnl: np.ndarray, lambda_: float = 1.0) -> float:
    """Entropic risk measure at risk-aversion coefficient lambda_.

    Defined as:
        rho(X) = (1 / lambda_) * log( E[ exp(-lambda_ * X) ] )

    where X is the P&L random variable.  This is also called the exponential
    premium principle or the certainty equivalent under exponential utility.

    Properties:
    - Convex: rho(t*X + (1-t)*Y) <= t*rho(X) + (1-t)*rho(Y).
    - Monotone: X >= Y a.s. => rho(X) <= rho(Y).
    - Translation-invariant: rho(X + c) = rho(X) - c.
    - As lambda_ -> 0, converges to -E[X] (mean loss).
    - As lambda_ -> inf, converges to ess sup(-X) (worst case).

    For X ~ N(mu, sigma^2):
        rho(X) = -mu + (lambda_ * sigma^2) / 2

    Numerical stability: the log-sum-exp trick avoids overflow in the exponential.
        log(E[exp(v)]) = logsumexp(v) - log(n)
    Applied here with v = -lambda_ * pnl.

    Parameters
    ----------
    pnl : np.ndarray
        Terminal P&L for each simulated path, shape (n_paths,).
    lambda_ : float
        Risk-aversion coefficient.  Must be > 0.  Larger values place more
        weight on extreme losses (more risk-averse).

    Returns
    -------
    float
        Entropic risk measure.  Larger values indicate more tail risk.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> pnl = rng.standard_normal(200_000)
    >>> abs(entropic_risk(pnl, lambda_=1.0) - 0.5) < 0.01  # N(0,1): rho = 0.5
    True
    """
    if lambda_ <= 0.0:
        raise ValueError(f"lambda_ must be positive, got {lambda_}")

    pnl_arr = np.asarray(pnl, dtype=float)
    n = len(pnl_arr)
    v = -lambda_ * pnl_arr
    log_mean_exp = logsumexp(v) - math.log(n)
    return float(log_mean_exp / lambda_)
