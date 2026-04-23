"""Compatibility shim — re-exports from src/neural/train.py.

Existing code imports train (and internal helpers) from src.train.
This shim preserves those imports while the canonical implementation lives at
src/neural/train.py.

New code should import from src.neural.train directly.
"""

from src.neural.train import _compute_pnl, _torch_cvar, train

__all__ = ["train", "_torch_cvar", "_compute_pnl"]
