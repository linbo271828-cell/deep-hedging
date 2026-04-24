"""Experiment 02: Classical delta hedger P&L distribution.

Shows the terminal P&L distribution of Black-Scholes delta-hedging under:
  - Zero transaction costs
  - 10 bps (0.001) proportional transaction costs

Paths are generated under the risk-neutral measure (mu = r) so that the
no-cost hedger's expected P&L is near zero, consistent with the no-arbitrage
pricing identity.

Outputs:
    results/plots/02_pnl_distribution.png

Prints to stdout:
    Mean, std, CVaR(95%) for both strategies, and the cost drag.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

import src.black_scholes as bs
from src.config import MarketConfig
from src.hedger import classical_delta_pnl
from src.market import GBM, time_grid
from src.risk_measures import cvar

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_PATHS = 20_000
COST_RATE_WITH = 0.001  # 10 bps
ALPHA = 0.95
PLOT_PATH = "results/plots/02_pnl_distribution.png"


def main() -> None:
    config = MarketConfig()

    # Risk-neutral GBM: mu = r so discounted price is a martingale.
    gbm_rn = GBM(s0=config.s0, mu=config.r, sigma=config.sigma)
    paths = gbm_rn.sample_paths(
        n_paths=N_PATHS, n_steps=config.n_steps, T=config.T, seed=SEED
    )
    tgrid = time_grid(n_steps=config.n_steps, T=config.T)

    # Initial option price (Black-Scholes, risk-neutral)
    initial_option_price = float(
        bs.call_price(config.s0, config.k, config.r, config.sigma, config.T)
    )

    # ---- Run both strategies ----
    pnl_no_cost = classical_delta_pnl(
        paths=paths,
        time_grid=tgrid,
        k=config.k,
        r=config.r,
        sigma=config.sigma,
        cost_rate=0.0,
        initial_option_price=initial_option_price,
    )

    pnl_with_cost = classical_delta_pnl(
        paths=paths,
        time_grid=tgrid,
        k=config.k,
        r=config.r,
        sigma=config.sigma,
        cost_rate=COST_RATE_WITH,
        initial_option_price=initial_option_price,
    )

    # ---- Statistics ----
    mean_nc = float(pnl_no_cost.mean())
    std_nc = float(pnl_no_cost.std(ddof=1))
    cvar_nc = cvar(pnl_no_cost, alpha=ALPHA)

    mean_wc = float(pnl_with_cost.mean())
    std_wc = float(pnl_with_cost.std(ddof=1))
    cvar_wc = cvar(pnl_with_cost, alpha=ALPHA)

    cost_drag = mean_nc - mean_wc  # positive = costs reduced the mean

    bps_label = int(round(COST_RATE_WITH * 10_000))

    print("=" * 60)
    print("Experiment 02 — Classical Delta Hedger P&L")
    print(f"  n_paths={N_PATHS:,}   n_steps={config.n_steps}"
          f"   T={config.T}   sigma={config.sigma}   r={config.r}")
    print(f"  Initial call price = {initial_option_price:.4f}")
    print("-" * 60)
    print(f"  No cost   — mean={mean_nc:+.4f}  std={std_nc:.4f}  CVaR({int(ALPHA*100)}%)={cvar_nc:.4f}")
    print(f"  {bps_label} bps  — mean={mean_wc:+.4f}  std={std_wc:.4f}  CVaR({int(ALPHA*100)}%)={cvar_wc:.4f}")
    print(f"  Cost drag (mean P&L reduction) = {cost_drag:.4f}")
    print(f"  Saved plot to: {PLOT_PATH}")
    print("=" * 60)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 5))

    # Shared bin range spanning both distributions
    lo = min(pnl_no_cost.min(), pnl_with_cost.min())
    hi = max(pnl_no_cost.max(), pnl_with_cost.max())
    bins = np.linspace(lo, hi, 80)

    ax.hist(
        pnl_no_cost,
        bins=bins,
        alpha=0.6,
        color="steelblue",
        label=f"No cost  (mean={mean_nc:+.3f}, CVaR={cvar_nc:.3f})",
    )
    ax.hist(
        pnl_with_cost,
        bins=bins,
        alpha=0.6,
        color="darkorange",
        label=f"{bps_label} bps   (mean={mean_wc:+.3f}, CVaR={cvar_wc:.3f})",
    )

    # Vertical dashed lines for means
    ax.axvline(mean_nc, color="steelblue", linestyle="--", linewidth=1.5, label=f"Mean no cost = {mean_nc:+.3f}")
    ax.axvline(mean_wc, color="darkorange", linestyle="--", linewidth=1.5, label=f"Mean {bps_label} bps = {mean_wc:+.3f}")

    ax.set_xlabel("Terminal P&L", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Classical Delta Hedger P&L: No Cost vs {bps_label} bps", fontsize=13)
    ax.legend(fontsize=9)
    fig.tight_layout()

    os.makedirs("plots", exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
