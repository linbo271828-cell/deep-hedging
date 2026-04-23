"""Experiment registry — STUB, not yet implemented.

Lane E (experiment system) owns this file.
Implement in Phase A after the baseline migration is complete.

The registry maps experiment names to ExperimentConfig instances.
Named experiments are the canonical way to ensure reproducibility:
  - every run has a name
  - that name maps to exactly one config
  - configs include seeds

Design principle: the registry is a dict-of-configs, not a plugin framework.
Keep it simple.

Target public interface:
    REGISTRY: dict[str, ExperimentConfig]  — all named experiment configs
    get_config(name: str) -> ExperimentConfig
    list_experiments() -> list[str]
"""

from __future__ import annotations

from src.experiments.configs import (
    EvaluationConfig,
    ExperimentConfig,
    HedgeUniverseConfig,
    MarketConfig,
    PayoffConfig,
    TrainingConfig,
)

# ---------------------------------------------------------------------------
# Phase A — baseline migration: reproduce v0.1 results
# ---------------------------------------------------------------------------

_BASELINE_MARKET = MarketConfig()
_BASELINE_PAYOFF = PayoffConfig(payoff_type="call", k=100.0)
_BASELINE_UNIVERSE = HedgeUniverseConfig(universe_type="stock_only")
_BASELINE_EVAL = EvaluationConfig()

REGISTRY: dict[str, ExperimentConfig] = {
    "gbm_call_no_cost": ExperimentConfig(
        name="gbm_call_no_cost",
        market=_BASELINE_MARKET,
        payoff=_BASELINE_PAYOFF,
        hedge_universe=_BASELINE_UNIVERSE,
        training=TrainingConfig(cost_rate=0.0, seed=42),
        evaluation=_BASELINE_EVAL,
    ),
    "gbm_call_5bps": ExperimentConfig(
        name="gbm_call_5bps",
        market=_BASELINE_MARKET,
        payoff=_BASELINE_PAYOFF,
        hedge_universe=_BASELINE_UNIVERSE,
        training=TrainingConfig(cost_rate=0.0005, seed=43),
        evaluation=_BASELINE_EVAL,
    ),
    # Phase B — multi-payoff (stubs; configs can be registered once payoffs are implemented)
    # Phase C — Heston (stubs)
    # Phase D — stock+option universe (stubs)
    # Phase E — ablations (stubs)
}


def get_config(name: str) -> ExperimentConfig:
    """Retrieve a named experiment config from the registry.

    Raises KeyError with helpful message if name is not found.
    """
    if name not in REGISTRY:
        available = sorted(REGISTRY.keys())
        raise KeyError(
            f"Experiment '{name}' not found. Available: {available}"
        )
    return REGISTRY[name]


def list_experiments() -> list[str]:
    """Return sorted list of all registered experiment names."""
    return sorted(REGISTRY.keys())
