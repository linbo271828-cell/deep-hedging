"""Compatibility shim — re-exports from src/risk/.

Existing code imports cvar and entropic_risk from src.risk_measures.
This shim preserves those imports while the canonical implementations live at:
  src/risk/cvar.py        (cvar)
  src/risk/entropic.py    (entropic_risk)

New code should import from src.risk.cvar and src.risk.entropic directly.
"""

from src.risk.cvar import cvar
from src.risk.entropic import entropic_risk

__all__ = ["cvar", "entropic_risk"]
