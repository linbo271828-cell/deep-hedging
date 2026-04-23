"""Compatibility shim — re-exports from src/experiments/configs.py.

Existing code imports MarketConfig and TrainingConfig from src.config.
This shim preserves those imports while the canonical (expanded) configs
live at src/experiments/configs.py.

New code should import from src.experiments.configs directly.
"""

from src.experiments.configs import MarketConfig, TrainingConfig

__all__ = ["MarketConfig", "TrainingConfig"]
