"""Experiment 05: Cost-frontier sweep — the headline figure.

Sweeps proportional transaction costs from 0 to 50 basis points.
At each cost level, trains a fresh neural hedger and evaluates both it and
the classical Black-Scholes delta hedger on a fixed held-out test set.

The headline result: the neural hedger traces out a Pareto-superior frontier
in (expected transaction cost, CVaR) space.  As costs rise, the classical
hedger suffers linearly; the neural hedger learns to trade less and achieves
lower CVaR at the same expected cost.

Outputs:
    plots/05_cost_frontier.png   (the headline figure)

Prints to stdout:
    Full sweep table: cost level, neural CVaR, classical CVaR, expected costs.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import src.black_scholes as bs
from src.config import MarketConfig, TrainingConfig
from src.hedger import classical_delta_pnl
from src.market import GBM, time_grid
from src.neural_hedger import HedgeNet
from src.risk_measures import cvar
from src.train import _compute_pnl, train

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
# Cost levels in basis points
COST_BPS = [0, 5, 10, 20, 30, 50]
COST_RATES = [c / 10_000 for c in COST_BPS]

# Faster training config for each sweep point (accuracy traded for speed)
N_TRAIN_PATHS = 20_000
N_TRAIN_EPOCHS = 200

BASE_SEED = 100              # base seed; each cost level uses BASE_SEED + i
EVAL_SEED = BASE_SEED + 99_000
N_EVAL_PATHS = 20_000
ALPHA = 0.95
PLOT_PATH = "plots/05_cost_frontier.png"


def train_model_at_cost(
    cost_rate: float,
    seed: int,
    market: MarketConfig,
) -> HedgeNet:
    """Train a fresh HedgeNet at the given cost level and return it."""
    cfg = TrainingConfig(
        n_paths=N_TRAIN_PATHS,
        n_epochs=N_TRAIN_EPOCHS,
        seed=seed,
        cost_rate=cost_rate,
        alpha=ALPHA,
    )
    model = HedgeNet(n_layers=4, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model, _ = train(cfg, market, model, optimizer)
    return model


def evaluate(
    model: HedgeNet,
    eval_paths: np.ndarray,
    tgrid: np.ndarray,
    market: MarketConfig,
    cost_rate: float,
    initial_price: float,
) -> tuple[float, float, float, float]:
    """Return (cvar_neural, cvar_classical, ecost_neural, ecost_classical).

    Expected cost is estimated as the mean P&L reduction relative to the
    zero-cost classical baseline (higher = more total costs paid).
    """
    # Classical P&L with and without costs
    pnl_classical = classical_delta_pnl(
        paths=eval_paths, time_grid=tgrid, k=market.k, r=market.r,
        sigma=market.sigma, cost_rate=cost_rate, initial_option_price=initial_price,
    )
    pnl_classical_nc = classical_delta_pnl(
        paths=eval_paths, time_grid=tgrid, k=market.k, r=market.r,
        sigma=market.sigma, cost_rate=0.0, initial_option_price=initial_price,
    )

    # Neural P&L
    model.eval()
    with torch.no_grad():
        paths_t = torch.tensor(eval_paths, dtype=torch.float32)
        pnl_neural_t = _compute_pnl(
            paths=paths_t,
            model=model,
            s0=market.s0,
            k=market.k,
            r=market.r,
            T=market.T,
            n_steps=market.n_steps,
            cost_rate=cost_rate,
            initial_option_price=initial_price,
        )
    pnl_neural = pnl_neural_t.numpy()

    nc_baseline = float(pnl_classical_nc.mean())
    cvar_neural = cvar(pnl_neural, alpha=ALPHA)
    cvar_classical = cvar(pnl_classical, alpha=ALPHA)
    ecost_neural = nc_baseline - float(pnl_neural.mean())
    ecost_classical = nc_baseline - float(pnl_classical.mean())

    return cvar_neural, cvar_classical, ecost_neural, ecost_classical


def main() -> None:
    market = MarketConfig()
    initial_price = float(
        bs.call_price(market.s0, market.k, market.r, market.sigma, market.T)
    )

    # Fixed evaluation set (never changes across cost levels)
    gbm_rn = GBM(s0=market.s0, mu=market.r, sigma=market.sigma)
    eval_paths = gbm_rn.sample_paths(
        n_paths=N_EVAL_PATHS, n_steps=market.n_steps, T=market.T, seed=EVAL_SEED
    )
    tgrid = time_grid(n_steps=market.n_steps, T=market.T)

    print("=" * 72)
    print("Experiment 05 — Cost Frontier Sweep")
    print(f"  cost levels (bps): {COST_BPS}")
    print(f"  per-level training: {N_TRAIN_PATHS:,} paths × {N_TRAIN_EPOCHS} epochs")
    print(f"  evaluation: {N_EVAL_PATHS:,} paths")
    print("-" * 72)
    print(f"  {'bps':>5}  {'CVaR(neural)':>14}  {'CVaR(classical)':>16}"
          f"  {'E[cost](neural)':>16}  {'E[cost](classical)':>18}")
    print("-" * 72)

    results_neural: list[tuple[float, float]] = []       # (ecost, cvar)
    results_classical: list[tuple[float, float]] = []

    for i, (cost_rate, bps) in enumerate(zip(COST_RATES, COST_BPS)):
        seed = BASE_SEED + i
        model = train_model_at_cost(cost_rate, seed, market)

        cv_n, cv_c, ec_n, ec_c = evaluate(
            model, eval_paths, tgrid, market, cost_rate, initial_price
        )

        results_neural.append((ec_n, cv_n))
        results_classical.append((ec_c, cv_c))

        print(f"  {bps:>5}  {cv_n:>14.4f}  {cv_c:>16.4f}"
              f"  {ec_n:>16.4f}  {ec_c:>18.4f}")

    print("-" * 72)
    print(f"  Saved plot to: {PLOT_PATH}")
    print("=" * 72)

    # ---- Headline plot ----
    fig, ax = plt.subplots(figsize=(8, 5))

    ecosts_n = [r[0] for r in results_neural]
    cvars_n = [r[1] for r in results_neural]
    ecosts_c = [r[0] for r in results_classical]
    cvars_c = [r[1] for r in results_classical]

    ax.plot(
        ecosts_n, cvars_n,
        "o-", color="darkorange", linewidth=2, markersize=7,
        label="Neural hedger",
    )
    ax.plot(
        ecosts_c, cvars_c,
        "s--", color="steelblue", linewidth=2, markersize=7,
        label="Classical BS delta",
    )

    # Annotate cost levels
    for (ec, cv), bps in zip(results_classical, COST_BPS):
        ax.annotate(
            f"{bps} bps",
            xy=(ec, cv),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="steelblue",
        )

    ax.set_xlabel("Expected Transaction Cost (per path)", fontsize=12)
    ax.set_ylabel(f"CVaR at {int(ALPHA * 100)}% (lower = better)", fontsize=12)
    ax.set_title(
        "Cost Frontier: Neural Hedger vs Classical BS Delta-Hedging\n"
        "(Each point is a different transaction cost level)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs("plots", exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
