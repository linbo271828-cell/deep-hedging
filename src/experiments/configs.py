"""Experiment configuration dataclasses for Deep Hedging Lab.

All seeds, model dimensions, market parameters, and experiment settings live here.
Experiment scripts and training loops import from this module only; they never
hardcode values inline.

HIGH-RISK SHARED FILE.
Rules:
  - Do not remove or rename fields without updating ALL callers.
  - Add new fields with defaults to avoid breaking existing code.
  - Seeds live here. Never hardcode seeds in experiments.
  - All configs are frozen dataclasses (immutable after construction).

Config hierarchy:
  MarketConfig    : GBM market parameters
  HestonConfig    : Heston parameters (stub)
  PayoffConfig    : payoff type and contract parameters
  HedgeUniverseConfig : which instruments are tradable
  TrainingConfig  : neural hedger training hyperparameters
  EvaluationConfig: evaluation settings
  ExperimentConfig: umbrella config grouping all of the above
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Market models
# ---------------------------------------------------------------------------


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
class HestonConfig:
    """Heston stochastic volatility parameters — stub for Phase C.

    Feller condition (for variance to stay positive a.s.): 2*kappa*theta >= xi^2.
    With defaults: 2*2.0*0.04 = 0.16 >= 0.09 = 0.3^2. Satisfied.
    """

    s0: float = 100.0
    r: float = 0.02
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    T: float = 0.5
    n_steps: int = 50


# ---------------------------------------------------------------------------
# Payoffs
# ---------------------------------------------------------------------------

PayoffType = Literal["call", "put", "call_spread", "straddle"]


@dataclass(frozen=True)
class PayoffConfig:
    """Contract specification for the option being hedged.

    For spreads: k is the lower strike, k_hi is the upper strike.
    For calls, puts, straddles: only k is used.
    """

    payoff_type: PayoffType = "call"
    k: float = 100.0
    k_hi: float = 110.0


# ---------------------------------------------------------------------------
# Hedge universe
# ---------------------------------------------------------------------------

HedgeUniverseType = Literal["stock_only", "stock_plus_option"]


@dataclass(frozen=True)
class HedgeUniverseConfig:
    """Which instruments are available to the hedger.

    stock_only       : single risky asset (Phase A/B/C)
    stock_plus_option: stock + one liquid vanilla option (Phase D)

    For stock_plus_option, hedge_option_strike/maturity define the liquid option.
    """

    universe_type: HedgeUniverseType = "stock_only"
    hedge_option_strike: float = 100.0
    hedge_option_maturity: float = 0.5


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationConfig:
    """Settings for out-of-sample evaluation of a trained hedger."""

    n_paths: int = 100_000
    seed_offset: int = 1000
    alpha: float = 0.95
    lambda_: float = 1.0


# ---------------------------------------------------------------------------
# Umbrella experiment config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    """Single unified config for a full experiment run.

    Groups all sub-configs. Named experiments in the registry instantiate this.
    """

    name: str
    market: MarketConfig = field(default_factory=MarketConfig)
    payoff: PayoffConfig = field(default_factory=PayoffConfig)
    hedge_universe: HedgeUniverseConfig = field(default_factory=HedgeUniverseConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
