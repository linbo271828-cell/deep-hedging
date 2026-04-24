"""Training loop for the neural delta hedger.

End-to-end CVaR minimization over simulated risk-neutral GBM paths.
The P&L accounting mirrors src/hedging/pnl.hedge_pnl exactly, but uses PyTorch
tensors so gradients flow back through the hedge-ratio decisions to the
model parameters.

Key design decisions:
- The cash account is NOT detached between steps: gradients propagate through
  the entire cash trajectory so the model learns the downstream effect of each trade.
- The previous delta IS detached when used as an input feature: this avoids a
  second-order gradient chain across the recurrence.
- Loss is CVaR via torch.sort (sub-gradient differentiable).
- Each epoch uses fresh paths with seed = config.seed + epoch, ensuring
  reproducible but non-repeated training data.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.models.gbm import GBM
from src.neural.architectures import HedgeNet
from src.payoffs.european import call_price

if TYPE_CHECKING:
    from src.experiments.configs import MarketConfig, PayoffConfig, TrainingConfig


def _torch_cvar(pnl: torch.Tensor, alpha: float) -> torch.Tensor:
    """Differentiable CVaR via torch.sort.

    Mean of the worst (1-alpha) fraction of P&L outcomes.
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
    payoff_fn: object = None,
    delta_transform: object = None,
) -> torch.Tensor:
    """Differentiable P&L for a neural delta-hedging strategy.

    Mirrors src/hedging/pnl.hedge_pnl accounting exactly in PyTorch.

    payoff_fn: callable(s_T tensor) -> tensor.  None = European call.
    delta_transform: callable(model_output) -> hedge ratio.  None = identity.
    """
    n_paths = paths.shape[0]
    dt = T / n_steps
    compound = math.exp(r * dt)

    s_curr = paths[:, 0]
    tau_curr = T

    delta_prev = torch.zeros(n_paths, dtype=paths.dtype, device=paths.device)
    feats = torch.stack(
        [s_curr / s0, torch.full((n_paths,), tau_curr, dtype=paths.dtype), delta_prev],
        dim=1,
    )
    raw = model(feats)
    delta = delta_transform(raw) if delta_transform is not None else raw

    cost = torch.abs(delta) * s_curr * cost_rate
    cash = initial_option_price - delta * s_curr - cost
    cash = cash * compound

    for i in range(1, n_steps):
        s_curr = paths[:, i]
        tau_curr = T - i * dt

        feats = torch.stack(
            [
                s_curr / s0,
                torch.full((n_paths,), tau_curr, dtype=paths.dtype),
                delta.detach(),
            ],
            dim=1,
        )
        raw_new = model(feats)
        delta_new = delta_transform(raw_new) if delta_transform is not None else raw_new

        trade = delta_new - delta
        cost = torch.abs(trade) * s_curr * cost_rate
        cash = cash - trade * s_curr - cost
        delta = delta_new
        cash = cash * compound

    s_T = paths[:, n_steps]
    liquidation_cost = torch.abs(delta) * s_T * cost_rate
    cash = cash + delta * s_T - liquidation_cost

    if payoff_fn is not None:
        payoff = torch.tensor(
            payoff_fn(s_T.detach().numpy()), dtype=torch.float32, device=s_T.device
        )
    else:
        payoff = torch.clamp(s_T - k, min=0.0)
    cash = cash - payoff

    return cash


def train(
    config: "TrainingConfig",
    market_config: "MarketConfig",
    model: HedgeNet,
    optimizer: torch.optim.Optimizer,
    payoff_config: "PayoffConfig | None" = None,
) -> tuple[HedgeNet, list[float]]:
    """Train HedgeNet by minimizing CVaR of terminal hedging P&L.

    Generates fresh risk-neutral GBM paths each epoch (seed = config.seed + epoch).

    Parameters
    ----------
    config:
        Training hyperparameters.
    market_config:
        Market parameters.
    model:
        HedgeNet instance, modified in place via optimizer.
    optimizer:
        Pre-constructed torch optimizer for model.parameters().
    payoff_config:
        Contract specification.  If None, defaults to European call (backward compat).

    Returns
    -------
    tuple[HedgeNet, list[float]]
        (trained model, per-epoch CVaR loss history)
    """
    torch.manual_seed(config.seed)

    gbm = GBM(s0=market_config.s0, mu=market_config.r, sigma=market_config.sigma)

    if payoff_config is not None and payoff_config.payoff_type != "call":
        from src.payoffs.dispatch import get_payoff_spec
        spec = get_payoff_spec(payoff_config, market_config)
        initial_option_price = spec.initial_option_price
        payoff_fn = spec.payoff_fn
        delta_transform = spec.delta_transform
    else:
        initial_option_price = float(
            call_price(
                market_config.s0,
                market_config.k,
                market_config.r,
                market_config.sigma,
                market_config.T,
            )
        )
        payoff_fn = None
        delta_transform = None

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
            payoff_fn=payoff_fn,
            delta_transform=delta_transform,
        )

        loss = _torch_cvar(pnl, alpha=config.alpha)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_history.append(loss.item())

    return model, loss_history
