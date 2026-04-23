"""Neural network architecture definitions for Deep Hedging Lab.

v1 architecture: HedgeNet — feedforward network, sigmoid output in [0,1].
Future architectures (recurrent, attention, position-aware) are out of scope for v1.

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
