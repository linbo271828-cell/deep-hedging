"""Heston stochastic-volatility model — STUB, not yet implemented.

Lane A (market models) owns this file.
Implement after the GBM baseline is migrated and tests pass.

Target interface (mirrors GBM.sample_paths):
    Heston.sample_paths(n_paths, n_steps, T, seed) -> np.ndarray (n_paths, n_steps+1)

The returned array is the *stock price* path only.
A companion method will return the *variance* path for Heston-aware feature builders.

Planned simulation method: Broadie-Kaya exact simulation or Andersen QE scheme.
The Euler-Maruyama scheme is acceptable for a first pass but must be documented as
approximate (unlike GBM where the exact solution is trivial).

Parameter reference:
    v0    : initial variance (typical: 0.04, i.e. sigma_0 = 20%)
    kappa : mean-reversion speed (typical: 2.0)
    theta : long-run variance (typical: 0.04)
    xi    : vol-of-vol (typical: 0.3)
    rho   : correlation between stock and variance Brownians (typical: -0.7)

Feller condition for well-posedness: 2 * kappa * theta >= xi**2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HestonParams:
    """Heston model parameters with typical market-calibrated defaults."""

    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7

    def __post_init__(self) -> None:
        if self.v0 <= 0:
            raise ValueError(f"v0 must be positive, got {self.v0}")
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.theta <= 0:
            raise ValueError(f"theta must be positive, got {self.theta}")
        if self.xi <= 0:
            raise ValueError(f"xi must be positive, got {self.xi}")
        if not (-1.0 <= self.rho <= 1.0):
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")

    @property
    def feller_satisfied(self) -> bool:
        """True iff 2*kappa*theta >= xi^2 (variance process stays positive a.s.)."""
        return 2.0 * self.kappa * self.theta >= self.xi**2


@dataclass(frozen=True)
class Heston:
    """Heston stochastic volatility model — NOT YET IMPLEMENTED.

    Placeholder that raises NotImplementedError to block accidental use.
    Replace this class body with the actual simulation when Lane A implements it.
    """

    s0: float
    r: float
    params: HestonParams

    def sample_paths(
        self, n_paths: int, n_steps: int, T: float, seed: int
    ) -> np.ndarray:
        """Return stock price paths, shape (n_paths, n_steps + 1).

        STUB — raises NotImplementedError.
        """
        raise NotImplementedError(
            "Heston.sample_paths is not yet implemented. "
            "See writeup/implementation_plan.md Phase 3 for the plan."
        )

    def sample_paths_with_variance(
        self, n_paths: int, n_steps: int, T: float, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (stock paths, variance paths), each shape (n_paths, n_steps + 1).

        STUB — raises NotImplementedError.
        The variance paths are needed by Heston-aware feature builders.
        """
        raise NotImplementedError(
            "Heston.sample_paths_with_variance is not yet implemented."
        )
