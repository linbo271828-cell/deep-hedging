"""Neural network hedge-ratio model for deep hedging.

Implements the feedforward network described in Buehler et al. (2019).
The network maps observable market features at each rebalancing step to a
hedge ratio (number of shares to hold), bounded in [0, 1] by a sigmoid output.

Architecture
------------
Linear(3, H) -> ReLU -> [Linear(H, H) -> ReLU] * (n_layers - 1) -> Linear(H, 1) -> Sigmoid

where H = hidden_dim and the total number of hidden layers is n_layers.

Input features (dimension 3 at each timestep t)
-------------------------------------------------
- S_t / S_0        : normalized stock price (≈ 1.0 at inception)
- tau_t            : time to maturity in years, in [0, T]
- current_delta    : previous period's hedge ratio (in [0, 1])

The inclusion of the previous delta as a feature allows the network to learn
transaction-cost-aware policies: large changes in delta are penalised by costs,
so the network can implicitly learn to trade off hedging error and turnover.

Output
------
Sigmoid of the final linear layer gives a hedge ratio in (0, 1), matching the
range of the Black-Scholes call delta.  No clipping is needed.

This module owns ONLY the network definition.
Training, P&L accounting, and risk measure computation live in separate modules.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HedgeNet(nn.Module):
    """Feedforward network mapping market features to a hedge ratio in [0, 1].

    Parameters
    ----------
    n_layers : int
        Total number of hidden layers (including the first projection layer).
        Must be >= 1.  Buehler et al. (2019) use 4 layers with 64 units.
    hidden_dim : int
        Number of units in each hidden layer.

    Examples
    --------
    >>> import torch
    >>> model = HedgeNet(n_layers=4, hidden_dim=64)
    >>> x = torch.randn(32, 3)   # batch of 32, 3 features
    >>> delta = model(x)
    >>> delta.shape
    torch.Size([32])
    >>> (delta >= 0).all() and (delta <= 1).all()
    True
    """

    def __init__(self, n_layers: int = 4, hidden_dim: int = 64) -> None:
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")

        super().__init__()

        layers: list[nn.Module] = []

        # Input projection: 3 features -> hidden_dim
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(nn.ReLU())

        # Additional hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer: hidden_dim -> 1, then sigmoid
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map market features to a hedge ratio.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 3).  Columns are:
            [0] S_t / S_0  — normalized stock price
            [1] tau_t      — time to maturity (years)
            [2] current_delta — previous hedge ratio in [0, 1]

        Returns
        -------
        torch.Tensor
            Shape (batch,).  Each element is the recommended hedge ratio
            (delta) for that path and timestep, in (0, 1).
        """
        return self.net(x).squeeze(-1)
