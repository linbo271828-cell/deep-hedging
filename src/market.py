"""Compatibility shim — re-exports from src/models/gbm.py.

Existing code (tests, experiments/01-05) imports from src.market.
This shim preserves those imports unchanged while the canonical implementation
lives at src/models/gbm.py.

New code should import from src.models.gbm directly.
"""

from src.models.gbm import GBM, MarketModel, time_grid

__all__ = ["GBM", "MarketModel", "time_grid"]
