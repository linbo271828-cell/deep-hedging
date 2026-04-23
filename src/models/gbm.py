"""Geometric Brownian Motion path simulator.

Exact-solution sampler (no Euler-Maruyama discretization error).

Math:
    S_{t+dt} = S_t * exp( (mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z ),  Z ~ N(0,1)

This is the canonical market model for Deep Hedging Lab v1.
The MarketModel protocol allows Heston (and future models) to share the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class MarketModel(Protocol):
    """Protocol satisfied by any Monte Carlo path simulator.

    Implementations must be deterministic given `seed`.
    """

    def sample_paths(
        self, n_paths: int, n_steps: int, T: float, seed: int
    ) -> np.ndarray:
        """Return asset-price paths, shape (n_paths, n_steps + 1).

        Index 0 is the initial price s0; index n_steps is the terminal price at T.
        """
        ...


@dataclass(frozen=True)
class GBM:
    """Geometric Brownian Motion under a constant drift.

    For risk-neutral training set mu = r (risk-free rate).
    For physical-measure validation set mu to the physical drift.
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
        """Sample GBM paths using the exact exponential solution.

        Returns shape (n_paths, n_steps + 1). Index 0 = s0.
        """
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
    """Uniform time grid for `n_steps` periods over [0, T], length n_steps+1."""
    return np.linspace(0.0, T, n_steps + 1)
