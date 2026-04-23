"""Experiment 01: GBM moment validation.

Verifies that the GBM simulator produces paths with the correct log-return
moments (mean and variance). Plots a histogram of terminal log-returns against
the theoretical normal density, plus a panel of sample paths.

Outputs:
    plots/01_gbm_moments.png

Prints to stdout:
    Empirical vs theoretical mean and variance with standard errors.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from src.config import MarketConfig
from src.market import GBM, time_grid

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_PATHS = 50_000
T_VAL = 1.0  # use T=1.0 for richer moment validation (not config.T=0.5)
N_SAMPLE_PATHS = 30  # paths to draw in the right panel
PLOT_PATH = "plots/01_gbm_moments.png"


def main() -> None:
    config = MarketConfig()

    # Generate paths using physical-measure mu (config.mu).
    gbm = GBM(s0=config.s0, mu=config.mu, sigma=config.sigma)
    paths = gbm.sample_paths(n_paths=N_PATHS, n_steps=config.n_steps, T=T_VAL, seed=SEED)
    tgrid = time_grid(n_steps=config.n_steps, T=T_VAL)

    # ---- Terminal log-returns ----
    log_returns = np.log(paths[:, -1] / config.s0)  # shape (N_PATHS,)

    # Theoretical moments
    theo_mean = gbm.log_return_mean(T_VAL)
    theo_var = gbm.log_return_variance(T_VAL)
    theo_std = np.sqrt(theo_var)

    # Empirical moments
    emp_mean = float(log_returns.mean())
    emp_var = float(log_returns.var(ddof=1))

    # Standard errors
    se_mean = float(log_returns.std(ddof=1) / np.sqrt(N_PATHS))
    # SE of sample variance: sqrt(2/(n-1)) * s^2
    se_var = float(emp_var * np.sqrt(2.0 / (N_PATHS - 1)))

    # Z-scores (how many SEs from theoretical)
    z_mean = (emp_mean - theo_mean) / se_mean
    z_var = (emp_var - theo_var) / se_var

    pass_mean = abs(z_mean) < 3.0
    pass_var = abs(z_var) < 3.0

    print("=" * 60)
    print("Experiment 01 — GBM Moment Validation")
    print(f"  n_paths={N_PATHS:,}   T={T_VAL}   mu={config.mu}   sigma={config.sigma}")
    print("-" * 60)
    print(f"  Log-return mean  : empirical={emp_mean:+.6f}  theoretical={theo_mean:+.6f}"
          f"   SE={se_mean:.6f}   z={z_mean:+.2f}  {'PASS' if pass_mean else 'FAIL'}")
    print(f"  Log-return var   : empirical={emp_var:.6f}   theoretical={theo_var:.6f}"
          f"   SE={se_var:.6f}   z={z_var:+.2f}  {'PASS' if pass_var else 'FAIL'}")
    print(f"  Saved plot to    : {PLOT_PATH}")
    print("=" * 60)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: histogram of terminal log-returns vs theoretical normal PDF
    ax = axes[0]
    x_lo = theo_mean - 4.5 * theo_std
    x_hi = theo_mean + 4.5 * theo_std
    x_grid = np.linspace(x_lo, x_hi, 500)
    pdf_vals = norm.pdf(x_grid, loc=theo_mean, scale=theo_std)

    ax.hist(
        log_returns,
        bins=80,
        density=True,
        color="steelblue",
        alpha=0.65,
        label=f"Empirical (n={N_PATHS:,})",
    )
    ax.plot(
        x_grid,
        pdf_vals,
        color="firebrick",
        linewidth=2.0,
        label=r"Theoretical $\mathcal{N}(\mu_{{log}},\,\sigma^2 T)$",
    )
    ax.axvline(emp_mean, color="steelblue", linestyle="--", linewidth=1.2, label=f"Emp mean = {emp_mean:.4f}")
    ax.axvline(theo_mean, color="firebrick", linestyle=":", linewidth=1.2, label=f"Theo mean = {theo_mean:.4f}")
    ax.set_xlabel(r"$\log(S_T / S_0)$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Terminal Log-Return Distribution", fontsize=13)
    ax.legend(fontsize=9)

    # Right panel: a sample of individual paths
    ax2 = axes[1]
    rng_idx = np.random.default_rng(SEED + 1)
    sample_idx = rng_idx.choice(N_PATHS, size=N_SAMPLE_PATHS, replace=False)
    for idx in sample_idx:
        ax2.plot(tgrid, paths[idx], color="steelblue", alpha=0.35, linewidth=0.8)
    # Overlay the mean path: E[S_t] = S0 * exp(mu * t)
    mean_path = config.s0 * np.exp(config.mu * tgrid)
    ax2.plot(tgrid, mean_path, color="firebrick", linewidth=2.0, label=r"$E[S_t]=S_0 e^{\mu t}$")
    ax2.set_xlabel("Time $t$", fontsize=12)
    ax2.set_ylabel("Asset price $S_t$", fontsize=12)
    ax2.set_title(f"Sample GBM Paths (n={N_SAMPLE_PATHS})", fontsize=13)
    ax2.legend(fontsize=9)

    fig.suptitle(
        f"GBM Validation  |  $\\mu={config.mu}$, $\\sigma={config.sigma}$, $T={T_VAL}$, "
        f"$S_0={config.s0}$",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()

    os.makedirs("plots", exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
