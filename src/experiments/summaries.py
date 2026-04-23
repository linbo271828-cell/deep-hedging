"""Experiment summary row schema.

SummaryRow is the canonical unit of output from a completed experiment.
A collection of SummaryRows forms the summary CSV / JSON for reporting.

Lane E (experiment system) owns this file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SummaryRow:
    """One row of experiment results for a single (config, strategy, seed) triple.

    Fields follow spec.md §9 (core metrics).
    All floats; None means the metric was not computed in this run.
    """

    experiment_name: str
    strategy: str
    seed: int
    cost_rate: float
    mean_pnl: float
    std_pnl: float
    cvar_95: float
    entropic_risk: float
    expected_cost: float
    turnover: float
    n_paths: int
    market_model: str = "gbm"
    payoff_type: str = "call"
    hedge_universe: str = "stock_only"
    n_epochs: int | None = None
    hidden_dim: int | None = None
    n_layers: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dict for CSV/JSON export."""
        return asdict(self)
