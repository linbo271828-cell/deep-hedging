"""Training loop for the neural delta hedger.

End-to-end CVaR minimization over simulated risk-neutral GBM paths.
The P&L accounting mirrors hedger.hedge_pnl exactly, but uses PyTorch
tensors so gradients flow back through the hedge-ratio decisions to the
model parameters.

Key design decisions:
- The cash account is NOT detached between steps: gradients flow through the
  entire cash trajectory, so the model learns the downstream effect of each trade.
- The previous delta IS detached when used as an input feature, avoiding
  a second-order gradient chain through the delta sequence.
- Loss is CVaR implemented via torch.sort, which is sub-gradient differentiable.
- Each epoch uses fresh GBM paths with seed = config.seed + epoch, so the
  training data is reproducible but not repeated across epochs.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from src import black_scholes as bs
from src.market import GBM
from src.neural_hedger import HedgeNet

if TYPE_CHECKING:
    from src.config import MarketConfig, TrainingConfig


def _torch_cvar(pnl: torch.Tensor, alpha: float) -> torch.Tensor:
    """Differentiable CVaR via torch.sort.

    Computes the mean of the worst (1-alpha) fraction of P&L outcomes.
    Larger return value = worse tail = higher loss for the optimizer.

    Uses sub-gradient of the sorted order (gradients exist almost everywhere).
    """
    n = pnl.shape[0]
    k = max(1, math.ceil((1.0 - alpha) * n))
    losses = -pnl
    sorted_losses, _ = torch.sort(losses, descending=True)
    return sorted_losses[:k].mean()


def _compute_pnl(
    paths: torch.Tensor,
    model: HedgeNet,
    s0: float,
    k: float,
    r: float,
    T: float,
    n_steps: int,
    cost_rate: float,
    initial_option_price: float,
) -> torch.Tensor:
    """Differentiable P&L for a neural delta-hedging strategy on a short call.

    Mirrors the accounting in hedger.hedge_pnl:
      t=0   : receive option premium; buy delta_0 shares; pay proportional cost.
      t=1..n-1: rebalance to new delta; pay cost on |trade| * S_t.
              Cash earns continuous interest exp(r * dt) each period.
      t=T   : liquidate position; settle call payoff max(S_T - K, 0).

    Gradients propagate through the cash account across all timesteps.
    The previous delta is detached when used as an input feature to avoid
    a complex second-order dependency chain across the recurrence.

    Parameters
    ----------
    paths:
        Asset prices, shape (n_paths, n_steps+1), dtype float32.
    model:
        HedgeNet mapping features (S_t/S_0, tau_t, prev_delta) -> delta in [0,1].
    s0, k, r, T, n_steps, cost_rate, initial_option_price:
        Market and contract parameters matching those in hedger.py.

    Returns
    -------
    torch.Tensor
        Terminal P&L per path, shape (n_paths,).
    """
    n_paths = paths.shape[0]
    dt = T / n_steps
    compound = math.exp(r * dt)

    # ---- t = 0 : open position ----
    s_curr = paths[:, 0]
    tau_curr = T

    delta_prev = torch.zeros(n_paths, dtype=paths.dtype, device=paths.device)
    feats = torch.stack(
        [s_curr / s0, torch.full((n_paths,), tau_curr, dtype=paths.dtype), delta_prev],
        dim=1,
    )
    delta = model(feats)  # (n_paths,) in [0, 1]

    cost = delta * s_curr * cost_rate
    cash = initial_option_price - delta * s_curr - cost
    cash = cash * compound

    # ---- t = 1 .. n_steps-1 : rebalance ----
    for i in range(1, n_steps):
        s_curr = paths[:, i]
        tau_curr = T - i * dt

        feats = torch.stack(
            [
                s_curr / s0,
                torch.full((n_paths,), tau_curr, dtype=paths.dtype),
                delta.detach(),  # previous delta is a feature, not a gradient source
            ],
            dim=1,
        )
        delta_new = model(feats)

        trade = delta_new - delta
        cost = torch.abs(trade) * s_curr * cost_rate
        cash = cash - trade * s_curr - cost
        delta = delta_new
        cash = cash * compound

    # ---- t = T : close position ----
    s_T = paths[:, n_steps]
    liquidation_cost = torch.abs(delta) * s_T * cost_rate
    cash = cash + delta * s_T - liquidation_cost

    payoff = torch.clamp(s_T - k, min=0.0)
    cash = cash - payoff

    return cash


def train(
    config: TrainingConfig,
    market_config: MarketConfig,
    model: HedgeNet,
    optimizer: torch.optim.Optimizer,
) -> tuple[HedgeNet, list[float]]:
    """Train HedgeNet by minimizing CVaR of terminal hedging P&L.

    Generates fresh risk-neutral GBM paths each epoch (seed = config.seed + epoch)
    so the effective training set is effectively unlimited and independent across
    epochs.  The risk-neutral drift (mu = r) ensures discounted prices are
    martingales, matching the Black-Scholes hedging framework.

    Parameters
    ----------
    config:
        Training hyperparameters: n_paths, n_epochs, lr, seed, cost_rate, alpha.
    market_config:
        Market parameters: s0, r, sigma, T, n_steps, k.
    model:
        HedgeNet instance. Modified in place via optimizer; also returned.
    optimizer:
        Torch optimizer pre-constructed for model.parameters().

    Returns
    -------
    tuple[HedgeNet, list[float]]
        (trained model, per-epoch CVaR loss history)
    """
    torch.manual_seed(config.seed)

    gbm = GBM(s0=market_config.s0, mu=market_config.r, sigma=market_config.sigma)
    initial_option_price = float(
        bs.call_price(
            market_config.s0, market_config.k, market_config.r,
            market_config.sigma, market_config.T,
        )
    )

    loss_history: list[float] = []

    for epoch in range(config.n_epochs):
        paths_np = gbm.sample_paths(
            n_paths=config.n_paths,
            n_steps=market_config.n_steps,
            T=market_config.T,
            seed=config.seed + epoch,
        )
        paths_t = torch.tensor(paths_np, dtype=torch.float32)

        pnl = _compute_pnl(
            paths=paths_t,
            model=model,
            s0=market_config.s0,
            k=market_config.k,
            r=market_config.r,
            T=market_config.T,
            n_steps=market_config.n_steps,
            cost_rate=config.cost_rate,
            initial_option_price=initial_option_price,
        )

        loss = _torch_cvar(pnl, alpha=config.alpha)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_history.append(loss.item())

    return model, loss_history
