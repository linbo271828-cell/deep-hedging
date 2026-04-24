"""Neural network architecture definitions for Deep Hedging Lab.

v1 architectures:
  HedgeNet      — single-instrument FFN, sigmoid output in [0,1].
  HedgeNetMulti — two-instrument FFN for stock + option hedge universe.

This module owns ONLY the network definitions.
Training, features, and evaluation live in separate modules.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HedgeNet(nn.Module):
    """Feedforward network mapping market features to a hedge ratio in [0, 1].

    Input features (dim 3 at each timestep):
        [0] S_t / S_0       normalized stock price
        [1] tau_t           time to maturity (years)
        [2] prev_delta      previous hedge ratio in [0, 1]

    Architecture:
        Linear(3, H) -> ReLU -> [Linear(H, H) -> ReLU] * (n_layers-1) -> Linear(H,1) -> Sigmoid

    The sigmoid output bounds the hedge ratio to (0, 1), matching the range
    of a European call delta.

    Parameters
    ----------
    n_layers:
        Number of hidden layers (>= 1). Default 4 follows Buehler et al. (2019).
    hidden_dim:
        Hidden units per layer. Default 64.

    Examples
    --------
    >>> model = HedgeNet(n_layers=4, hidden_dim=64)
    >>> x = torch.randn(32, 3)
    >>> delta = model(x)
    >>> delta.shape
    torch.Size([32])
    >>> bool((delta >= 0).all() and (delta <= 1).all())
    True
    """

    def __init__(self, n_layers: int = 4, hidden_dim: int = 64) -> None:
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")

        super().__init__()

        layers: list[nn.Module] = []
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map features to hedge ratio.

        Parameters
        ----------
        x:
            Shape (batch, 3). Columns: [S_t/S_0, tau_t, prev_delta].

        Returns
        -------
        torch.Tensor
            Shape (batch,). Hedge ratio in (0, 1).
        """
        return self.net(x).squeeze(-1)


class HedgeNetMulti(nn.Module):
    """Two-instrument feedforward hedger for stock + option hedge universe.

    Designed for the stock_plus_option universe where the hedger controls both
    the underlying stock position and a position in one liquid hedge option.

    Input features (dim 5 at each timestep):
        [0] S_t / S_0          normalized stock price
        [1] tau_t              time to maturity (years)
        [2] prev_delta_stock   previous stock hedge ratio
        [3] prev_delta_hedge   previous hedge-option position
        [4] V_hedge / V_hedge_0  normalized hedge-option price

    Output (dim 2), both sigmoid-bounded to (0, 1):
        [0] raw_stock_delta    — apply delta_transform for payoff-specific range
        [1] hedge_option_pos   — hedge-option units held ∈ (0, 1)

    Architecture:
        Linear(5, H) -> ReLU -> [Linear(H, H) -> ReLU] * (n_layers-1) -> Linear(H,2) -> Sigmoid

    Parameters
    ----------
    n_layers:
        Number of hidden layers (>= 1).
    hidden_dim:
        Hidden units per layer.
    """

    def __init__(self, n_layers: int = 4, hidden_dim: int = 64) -> None:
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")

        super().__init__()

        layers: list[nn.Module] = []
        layers.append(nn.Linear(5, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 2))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map 5 market features to two hedge positions.

        Parameters
        ----------
        x:
            Shape (batch, 5). See class docstring for column layout.

        Returns
        -------
        torch.Tensor
            Shape (batch, 2). Both values in (0, 1).
            Col 0: raw stock delta (apply delta_transform for payoff range).
            Col 1: hedge-option position.
        """
        return self.net(x)
