"""Feature builders for neural hedger inputs — partial stub.

The feature builder determines what information the neural hedger sees at each
timestep.  This is a key design dimension: the ablation study (Phase E, RQ5)
will compare feature sets.

v1 feature sets:
  1. base_features: (S_t/S_0, tau_t, prev_delta) — 3 features, implemented
  2. heston_features: base + (v_t,) — 4 features, STUB (requires Heston paths)

FeatureBuilder protocol: callable (state) -> np.ndarray or torch.Tensor of shape
(n_paths, n_features).

The base_features function is simple enough that it is currently inlined in
src/neural/train.py.  This module is the canonical home for feature logic once
the ablation experiments require multiple variants.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class FeatureBuilder(Protocol):
    """Protocol for feature-building callables."""

    def __call__(
        self,
        s_t: np.ndarray,
        s0: float,
        tau_t: float,
        prev_delta: np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        """Return feature matrix of shape (n_paths, n_features)."""
        ...


def base_features(
    s_t: np.ndarray,
    s0: float,
    tau_t: float,
    prev_delta: np.ndarray,
) -> np.ndarray:
    """3-feature input: (S_t/S_0, tau_t, prev_delta).

    This is the standard feature set used in v0.1 and the Phase A migration.

    Parameters
    ----------
    s_t:
        Current stock prices, shape (n_paths,).
    s0:
        Initial stock price (scalar).
    tau_t:
        Current time to maturity (scalar, same for all paths).
    prev_delta:
        Previous hedge ratio, shape (n_paths,), in [0, 1].

    Returns
    -------
    np.ndarray
        Shape (n_paths, 3).
    """
    n = len(s_t)
    return np.column_stack([
        s_t / s0,
        np.full(n, tau_t),
        prev_delta,
    ])


def heston_features(
    s_t: np.ndarray,
    s0: float,
    tau_t: float,
    prev_delta: np.ndarray,
    v_t: np.ndarray,
) -> np.ndarray:
    """4-feature input: (S_t/S_0, tau_t, prev_delta, v_t).

    STUB — the 4th feature (spot variance) requires Heston path simulation.
    Implement in Phase C after src/models/heston.py is complete.

    Parameters
    ----------
    v_t:
        Current instantaneous variance, shape (n_paths,).
    """
    raise NotImplementedError(
        "heston_features requires Heston path simulation. "
        "Implement in Phase C after src/models/heston.py is complete."
    )
