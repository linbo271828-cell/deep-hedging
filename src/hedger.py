"""Compatibility shim — re-exports from src/hedging/.

Existing code imports hedge_pnl and classical_delta_pnl from src.hedger.
This shim preserves those imports while the canonical implementations live at:
  src/hedging/pnl.py         (hedge_pnl)
  src/hedging/baseline.py    (classical_delta_pnl)

New code should import from src.hedging.pnl and src.hedging.baseline directly.
"""

from src.hedging.baseline import classical_delta_pnl
from src.hedging.pnl import hedge_pnl

__all__ = ["hedge_pnl", "classical_delta_pnl"]
