"""Sanity checks for the GBM simulator.

The claims we enforce:
  1. Determinism:  same seed -> byte-identical paths.
  2. Shape:        sample_paths returns (n_paths, n_steps + 1) and starts at s0.
  3. Moments:      empirical mean / variance of log(S_T / S_0) match the
                   closed-form GBM moments within a few standard errors.
  4. Invariants:   paths stay strictly positive (GBM cannot cross zero).
  5. Input checks: bad inputs raise ValueError.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.market import GBM, time_grid


def test_sample_paths_shape_and_start() -> None:
    gbm = GBM(s0=100.0, mu=0.05, sigma=0.2)
    paths = gbm.sample_paths(n_paths=256, n_steps=40, T=1.0, seed=0)
    assert paths.shape == (256, 41)
    assert np.allclose(paths[:, 0], 100.0)


def test_sample_paths_is_deterministic() -> None:
    gbm = GBM(s0=100.0, mu=0.05, sigma=0.2)
    a = gbm.sample_paths(n_paths=128, n_steps=20, T=1.0, seed=42)
    b = gbm.sample_paths(n_paths=128, n_steps=20, T=1.0, seed=42)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_produce_different_paths() -> None:
    gbm = GBM(s0=100.0, mu=0.05, sigma=0.2)
    a = gbm.sample_paths(n_paths=64, n_steps=10, T=1.0, seed=1)
    b = gbm.sample_paths(n_paths=64, n_steps=10, T=1.0, seed=2)
    assert not np.allclose(a, b)


def test_paths_are_strictly_positive() -> None:
    gbm = GBM(s0=1.0, mu=0.0, sigma=2.0)  # high vol stress
    paths = gbm.sample_paths(n_paths=1000, n_steps=500, T=2.0, seed=7)
    assert (paths > 0).all()


def test_log_return_moments_match_closed_form() -> None:
    """E[log(S_T/S_0)] and Var[log(S_T/S_0)] should match (mu - sigma^2/2)T and sigma^2 T.

    With 50k paths, the standard error on the mean is ~sigma*sqrt(T)/sqrt(N);
    we allow a 4-sigma tolerance which is essentially never violated.
    """
    mu, sigma, T = 0.08, 0.25, 1.0
    n_paths = 50_000
    gbm = GBM(s0=100.0, mu=mu, sigma=sigma)
    paths = gbm.sample_paths(n_paths=n_paths, n_steps=250, T=T, seed=123)

    log_rets = np.log(paths[:, -1] / paths[:, 0])
    emp_mean = log_rets.mean()
    emp_var = log_rets.var(ddof=1)

    theo_mean = gbm.log_return_mean(T)
    theo_var = gbm.log_return_variance(T)

    mean_se = np.sqrt(theo_var / n_paths)
    # Variance has SE ~ variance * sqrt(2 / (N-1)) for normal samples.
    var_se = theo_var * np.sqrt(2.0 / (n_paths - 1))

    assert abs(emp_mean - theo_mean) < 4 * mean_se, (
        f"mean off: emp={emp_mean:.6f} theo={theo_mean:.6f} se={mean_se:.6f}"
    )
    assert abs(emp_var - theo_var) < 4 * var_se, (
        f"var off: emp={emp_var:.6f} theo={theo_var:.6f} se={var_se:.6f}"
    )


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        GBM(s0=-1.0, mu=0.0, sigma=0.2)
    with pytest.raises(ValueError):
        GBM(s0=100.0, mu=0.0, sigma=-0.1)
    gbm = GBM(s0=100.0, mu=0.0, sigma=0.2)
    with pytest.raises(ValueError):
        gbm.sample_paths(n_paths=0, n_steps=10, T=1.0, seed=0)
    with pytest.raises(ValueError):
        gbm.sample_paths(n_paths=10, n_steps=0, T=1.0, seed=0)
    with pytest.raises(ValueError):
        gbm.sample_paths(n_paths=10, n_steps=10, T=0.0, seed=0)


def test_time_grid() -> None:
    grid = time_grid(n_steps=4, T=1.0)
    np.testing.assert_array_almost_equal(grid, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert len(grid) == 5
