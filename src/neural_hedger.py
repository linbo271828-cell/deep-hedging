"""Compatibility shim — re-exports from src/neural/architectures.py.

Existing code imports HedgeNet from src.neural_hedger.
This shim preserves that import while the canonical implementation lives at
src/neural/architectures.py.

New code should import from src.neural.architectures directly.
"""

from src.neural.architectures import HedgeNet

__all__ = ["HedgeNet"]
