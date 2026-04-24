"""Payoff dispatch — maps PayoffConfig to concrete callables.

Single source of truth for:
  - terminal payoff function (used at expiry in P&L accounting)
  - initial option price (premium received at t=0)
  - classical delta function (used by the baseline hedger)
  - delta transform (maps sigmoid network output to the correct hedge ratio range)

All four elements must be consistent for a given payoff type.

Delta transform design
----------------------
HedgeNet uses a sigmoid output bounded to (0, 1).  Payoffs whose natural
delta range differs require a linear remap before the output is used as a
hedge ratio:

  call       : delta ∈ [0, 1]    transform = identity
  put        : delta ∈ [−1, 0]   transform = x − 1
  call_spread: delta ∈ [0, 1]    transform = identity  (difference of two call deltas)
  straddle   : delta ∈ [−1, 1]   transform = 2x − 1

The transform is applied *inside* the training/evaluation P&L loop so gradients
still flow through the sigmoid naturally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

import numpy as np

from src.payoffs import european as bs
from src.payoffs.spreads import call_spread_payoff
from src.payoffs.straddles import straddle_payoff

if TYPE_CHECKING:
    from src.experiments.configs import MarketConfig, PayoffConfig


@dataclass
class PayoffSpec:
    """All callables needed to run a full training/evaluation/benchmark cycle."""

    payoff_fn: Callable[[np.ndarray], np.ndarray]
    initial_option_price: float
    classical_delta_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    delta_transform: Callable  # works on both np.ndarray and torch.Tensor
    classical_description: str


def get_payoff_spec(
    payoff_config: "PayoffConfig",
    market_config: "MarketConfig",
) -> PayoffSpec:
    """Return a PayoffSpec for the given payoff and market configuration.

    Parameters
    ----------
    payoff_config:
        Contract specification — payoff_type, strike k, optional k_hi.
    market_config:
        GBM market parameters — used to compute initial option price and
        to close over r, sigma for the classical delta function.

    Returns
    -------
    PayoffSpec
        Ready-to-use callables for training, evaluation, and baseline hedging.
    """
    mc = market_config
    pc = payoff_config

    if pc.payoff_type == "call":
        return PayoffSpec(
            payoff_fn=lambda s_T: np.asarray(bs.call_payoff(s_T, pc.k)),
            initial_option_price=float(
                bs.call_price(mc.s0, pc.k, mc.r, mc.sigma, mc.T)
            ),
            classical_delta_fn=lambda s, tau: np.asarray(
                bs.call_delta(s, pc.k, mc.r, mc.sigma, tau)
            ),
            delta_transform=lambda x: x,
            classical_description="Black-Scholes call delta N(d₁)",
        )

    elif pc.payoff_type == "put":
        return PayoffSpec(
            payoff_fn=lambda s_T: np.asarray(bs.put_payoff(s_T, pc.k)),
            initial_option_price=float(
                bs.put_price(mc.s0, pc.k, mc.r, mc.sigma, mc.T)
            ),
            classical_delta_fn=lambda s, tau: np.asarray(
                bs.put_delta(s, pc.k, mc.r, mc.sigma, tau)
            ),
            delta_transform=lambda x: x - 1.0,
            classical_description="Black-Scholes put delta N(d₁)−1",
        )

    elif pc.payoff_type == "call_spread":
        k_lo, k_hi = pc.k, pc.k_hi
        return PayoffSpec(
            payoff_fn=lambda s_T: np.asarray(call_spread_payoff(s_T, k_lo, k_hi)),
            initial_option_price=float(
                bs.call_price(mc.s0, k_lo, mc.r, mc.sigma, mc.T)
                - bs.call_price(mc.s0, k_hi, mc.r, mc.sigma, mc.T)
            ),
            classical_delta_fn=lambda s, tau: (
                np.asarray(bs.call_delta(s, k_lo, mc.r, mc.sigma, tau))
                - np.asarray(bs.call_delta(s, k_hi, mc.r, mc.sigma, tau))
            ),
            delta_transform=lambda x: x,
            classical_description=(
                f"Net BS delta: call_delta(K={k_lo})−call_delta(K={k_hi})"
            ),
        )

    elif pc.payoff_type == "straddle":
        return PayoffSpec(
            payoff_fn=lambda s_T: np.asarray(straddle_payoff(s_T, pc.k)),
            initial_option_price=float(
                bs.call_price(mc.s0, pc.k, mc.r, mc.sigma, mc.T)
                + bs.put_price(mc.s0, pc.k, mc.r, mc.sigma, mc.T)
            ),
            classical_delta_fn=lambda s, tau: (
                np.asarray(bs.call_delta(s, pc.k, mc.r, mc.sigma, tau))
                + np.asarray(bs.put_delta(s, pc.k, mc.r, mc.sigma, tau))
            ),
            delta_transform=lambda x: 2.0 * x - 1.0,
            classical_description="Net BS delta: call_delta+put_delta = 2·N(d₁)−1",
        )

    else:
        raise ValueError(
            f"Unknown payoff_type {pc.payoff_type!r}. "
            "Valid: 'call', 'put', 'call_spread', 'straddle'."
        )
