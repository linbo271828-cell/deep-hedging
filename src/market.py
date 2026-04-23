"""Market models for path simulation.

Day 1: geometric Brownian motion (GBM) exact-solution sampler.
Day 11 stretch: Heston stochastic volatility.

Math reference: writeup/math_reference.md §1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class MarketModel(Protocol):
    """Any asset-price model that can produce Monte Carlo paths.

    Implementations must be deterministic given `seed` so that experiments
    are reproducible and the sprint's verification checks pass.
    """

    def sample_paths(
        self, n_paths: int, n_steps: int, T: float, seed: int
    ) -> np.ndarray:
        """Return an array of shape (n_paths, n_steps + 1) of asset prices.

        Index 0 is `s0`; index n_steps is the terminal price at time T.
        """


@dataclass(frozen=True)
class GBM:
    """Geometric Brownian motion under the physical measure.

    SDE:  dS_t = mu * S_t * dt + sigma * S_t * dW_t

    We sample using the exact solution
        S_{t+dt} = S_t * exp( (mu - sigma^2 / 2) * dt + sigma * sqrt(dt) * Z )
    where Z ~ N(0, 1).  This is unbiased (no Euler-Maruyama discretization
    error) and vectorizes trivially.

    For option-pricing sanity checks (Monte Carlo vs. Black-Scholes closed
    form) use the risk-neutral GBM by setting `mu = r`.
    """

    s0: float
    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.s0 <= 0:
            raise ValueError(f"s0 must be positive, got {self.s0}")
        if self.sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")

    def sample_paths(
        self, n_paths: int, n_steps: int, T: float, seed: int
    ) -> np.ndarray:
        if n_paths <= 0 or n_steps <= 0 or T <= 0:
            raise ValueError("n_paths, n_steps, T must all be positive")

        dt = T / n_steps
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(size=(n_paths, n_steps))

        log_increments = (
            (self.mu - 0.5 * self.sigma**2) * dt
            + self.sigma * np.sqrt(dt) * z
        )
        cum_log = np.concatenate(
            [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1
        )
        return self.s0 * np.exp(cum_log)

    def log_return_mean(self, T: float) -> float:
        """E[log(S_T / S_0)] under the physical measure."""
        return (self.mu - 0.5 * self.sigma**2) * T

    def log_return_variance(self, T: float) -> float:
        """Var[log(S_T / S_0)]."""
        return self.sigma**2 * T


def time_grid(n_steps: int, T: float) -> np.ndarray:
    """Uniform time grid for `n_steps` periods over [0, T]."""
    return np.linspace(0.0, T, n_steps + 1)
