"""Experiment hyperparameters as frozen dataclasses.

All seeds, model dimensions, and market parameters live here.
Experiment scripts import and instantiate these; they never hardcode values inline.

This is a high-risk shared file. Do not remove or rename fields without updating all callers.
Add new fields with defaults to avoid breaking existing code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketConfig:
    """GBM market parameters and rebalancing grid."""

    s0: float = 100.0
    mu: float = 0.05
    r: float = 0.02
    sigma: float = 0.20
    T: float = 0.5
    n_steps: int = 50
    k: float = 100.0


@dataclass(frozen=True)
class TrainingConfig:
    """Neural hedger training hyperparameters."""

    n_paths: int = 50_000
    n_epochs: int = 200
    lr: float = 1e-3
    hidden_dim: int = 64
    n_layers: int = 4
    seed: int = 42
    cost_rate: float = 0.0
    alpha: float = 0.95
