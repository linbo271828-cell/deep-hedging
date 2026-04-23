"""Tests for src/risk_measures.py.

Philosophy: tests verify mathematical claims against known closed-form results.
No mocking. Deterministic samples via seeded RNG. Total runtime well under 5 s.

Closed-form references used:
- CVaR of N(0,1): phi(Phi^{-1}(alpha)) / (1 - alpha)   [standard result]
- Entropic risk of N(mu, sigma^2): -mu + lambda_*sigma^2/2
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from src.risk_measures import cvar, entropic_risk


# ---------------------------------------------------------------------------
# CVaR tests
# ---------------------------------------------------------------------------


def test_cvar_gaussian_known_value() -> None:
    """CVaR_0.95 of a standard Gaussian P&L should be ≈ 2.0627.

    For pnl ~ N(0, 1), the worst 5% of outcomes (most negative pnl) have
    expected value:
        CVaR_0.95 = phi(Phi^{-1}(0.95)) / (1 - 0.95) ≈ 2.0627
    where phi is the standard normal PDF and Phi is the CDF.
    """
    rng = np.random.default_rng(42)
    pnl = rng.standard_normal(200_000)

    alpha = 0.95
    analytic = norm.pdf(norm.ppf(alpha)) / (1.0 - alpha)  # ≈ 2.0627

    result = cvar(pnl, alpha=alpha)
    assert abs(result - analytic) < 0.01, (
        f"CVaR(N(0,1), 0.95): expected ≈ {analytic:.4f}, got {result:.4f}"
    )


def test_cvar_constant_is_loss() -> None:
    """CVaR of a constant-loss distribution should equal that loss.

    If every path loses exactly 5, the expected shortfall is 5.
    """
    pnl = np.full(1000, -5.0)
    result = cvar(pnl)
    np.testing.assert_allclose(result, 5.0, atol=1e-10)


def test_cvar_constant_positive_is_negative_gain() -> None:
    """CVaR of a constant-gain distribution returns a negative number.

    If every path gains 3, the 'expected loss in worst 5%' is -3
    (we're being paid, not losing).
    """
    pnl = np.full(1000, 3.0)
    result = cvar(pnl)
    np.testing.assert_allclose(result, -3.0, atol=1e-10)


def test_cvar_is_at_least_var() -> None:
    """CVaR >= VaR for any distribution (coherent risk measure property).

    VaR_alpha = the alpha-quantile of losses = -quantile(pnl, 1 - alpha).
    CVaR_alpha >= VaR_alpha always.
    """
    rng = np.random.default_rng(7)
    # Use a skewed distribution (exponential shifted to have losses)
    pnl = -rng.exponential(scale=2.0, size=50_000)

    alpha = 0.95
    var_alpha = float(np.quantile(-pnl, alpha))  # alpha-quantile of losses
    cvar_alpha = cvar(pnl, alpha=alpha)

    assert cvar_alpha >= var_alpha - 1e-10, (
        f"CVaR ({cvar_alpha:.4f}) should be >= VaR ({var_alpha:.4f})"
    )


def test_cvar_ordering() -> None:
    """Shifting all P&L down by 1 must strictly increase CVaR.

    pnl_b = pnl_a - 1 means uniformly worse outcomes.
    CVaR(pnl_b) = CVaR(pnl_a) + 1 by translation-invariance.
    """
    rng = np.random.default_rng(13)
    pnl_a = rng.standard_normal(10_000)
    pnl_b = pnl_a - 1.0

    assert cvar(pnl_b) > cvar(pnl_a), (
        f"Worse distribution must have higher CVaR: "
        f"cvar(pnl_b)={cvar(pnl_b):.4f}, cvar(pnl_a)={cvar(pnl_a):.4f}"
    )


def test_cvar_translation_invariance() -> None:
    """CVaR(pnl + c) = CVaR(pnl) - c  (translation-invariance of coherent measures)."""
    rng = np.random.default_rng(17)
    pnl = rng.standard_normal(10_000)
    c = 2.5
    np.testing.assert_allclose(cvar(pnl + c), cvar(pnl) - c, atol=1e-10)


def test_cvar_alpha_zero_is_mean_loss() -> None:
    """CVaR at alpha=0 is the mean loss (-mean(pnl))."""
    rng = np.random.default_rng(21)
    pnl = rng.standard_normal(5_000)
    np.testing.assert_allclose(cvar(pnl, alpha=0.0), -pnl.mean(), atol=1e-10)


def test_cvar_alpha_one_is_max_loss() -> None:
    """CVaR at alpha=1.0 is the maximum loss (-min(pnl))."""
    rng = np.random.default_rng(23)
    pnl = rng.standard_normal(5_000)
    np.testing.assert_allclose(cvar(pnl, alpha=1.0), -pnl.min(), atol=1e-10)


# ---------------------------------------------------------------------------
# Entropic risk tests
# ---------------------------------------------------------------------------


def test_entropic_risk_gaussian_known_value() -> None:
    """Entropic risk of N(0, 1) with lambda_=1 should be ≈ 0.5.

    For pnl ~ N(mu, sigma^2):
        entropic_risk(pnl, lambda_) = -mu + lambda_ * sigma^2 / 2
    For mu=0, sigma=1, lambda_=1: result = 0.5.
    """
    rng = np.random.default_rng(42)
    pnl = rng.standard_normal(200_000)

    result = entropic_risk(pnl, lambda_=1.0)
    analytic = 0.5  # -0 + 1.0 * 1.0 / 2

    assert abs(result - analytic) < 0.01, (
        f"Entropic risk of N(0,1), lambda=1: expected ≈ {analytic}, got {result:.4f}"
    )


def test_entropic_risk_gaussian_nonzero_mean() -> None:
    """Entropic risk of N(mu, sigma^2) = -mu + lambda_*sigma^2/2."""
    rng = np.random.default_rng(55)
    mu, sigma, lambda_ = 1.0, 0.5, 2.0
    pnl = mu + sigma * rng.standard_normal(200_000)

    analytic = -mu + lambda_ * sigma**2 / 2.0  # = -1 + 2*0.25 = -0.5
    result = entropic_risk(pnl, lambda_=lambda_)

    assert abs(result - analytic) < 0.02, (
        f"Entropic risk N({mu},{sigma}^2), lambda={lambda_}: "
        f"expected ≈ {analytic:.4f}, got {result:.4f}"
    )


def test_entropic_risk_convexity() -> None:
    """Entropic risk satisfies convexity.

    rho(0.5*X + 0.5*Y) <= 0.5*rho(X) + 0.5*rho(Y)
    """
    rng = np.random.default_rng(99)
    pnl_a = rng.standard_normal(10_000)
    pnl_b = rng.standard_normal(10_000) * 2.0 - 1.0  # different distribution

    lhs = entropic_risk(0.5 * pnl_a + 0.5 * pnl_b, lambda_=1.0)
    rhs = 0.5 * entropic_risk(pnl_a, lambda_=1.0) + 0.5 * entropic_risk(pnl_b, lambda_=1.0)

    assert lhs <= rhs + 1e-10, (
        f"Convexity violated: lhs={lhs:.6f} > rhs={rhs:.6f}"
    )


def test_entropic_risk_ordering() -> None:
    """Shifting P&L down by 1 must increase entropic risk by 1 (translation-invariance)."""
    rng = np.random.default_rng(37)
    pnl = rng.standard_normal(10_000)
    c = 1.5
    np.testing.assert_allclose(
        entropic_risk(pnl - c, lambda_=1.0),
        entropic_risk(pnl, lambda_=1.0) + c,
        atol=1e-10,
    )


def test_entropic_risk_increases_with_lambda() -> None:
    """For a risky distribution, higher lambda_ should give higher (or equal) risk.

    This follows from the interpretation: more risk-averse -> higher measured risk.
    For any pnl with variance > 0, increasing lambda_ increases the measure.
    """
    rng = np.random.default_rng(61)
    pnl = rng.standard_normal(10_000)  # zero mean, unit variance

    r1 = entropic_risk(pnl, lambda_=0.5)
    r2 = entropic_risk(pnl, lambda_=2.0)

    assert r2 > r1, (
        f"Higher lambda should give higher entropic risk: lambda=0.5 -> {r1:.4f}, "
        f"lambda=2.0 -> {r2:.4f}"
    )


def test_cvar_invalid_alpha() -> None:
    """CVaR with out-of-range alpha should raise ValueError."""
    pnl = np.zeros(100)
    with pytest.raises(ValueError):
        cvar(pnl, alpha=1.5)
    with pytest.raises(ValueError):
        cvar(pnl, alpha=-0.1)


def test_entropic_risk_invalid_lambda() -> None:
    """entropic_risk with non-positive lambda_ should raise ValueError."""
    pnl = np.zeros(100)
    with pytest.raises(ValueError):
        entropic_risk(pnl, lambda_=0.0)
    with pytest.raises(ValueError):
        entropic_risk(pnl, lambda_=-1.0)
