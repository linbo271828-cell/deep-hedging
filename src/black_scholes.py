"""Compatibility shim — re-exports from src/payoffs/european.py.

Existing code (tests, experiments) imports from src.black_scholes or
uses `from src import black_scholes as bs`.
This shim preserves those imports while the canonical implementation lives
at src/payoffs/european.py.

New code should import from src.payoffs.european directly.
"""

from src.payoffs.european import (
    _d1,
    _d2,
    call_delta,
    call_gamma,
    call_price,
    call_rho,
    call_theta,
    call_vega,
    put_price,
)

__all__ = [
    "_d1",
    "_d2",
    "call_price",
    "put_price",
    "call_delta",
    "call_gamma",
    "call_vega",
    "call_theta",
    "call_rho",
]
