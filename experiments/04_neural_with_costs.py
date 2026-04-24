"""Experiment 04: Neural hedger under transaction costs.

Trains the neural hedger with 5 bps (0.0005) proportional transaction costs
and compares the learned hedge ratio against:
  (a) Black-Scholes delta (the classical frictionless target), and
  (b) The zero-cost neural hedger (experiment 03 result).

The key result: with costs, the network learns a smoother, less reactive
policy that reduces turnover, accepting slightly more variance in exchange
for lower expected transaction costs.  The deviation from BS delta should be
largest near ATM, where delta changes most rapidly.

Prerequisite: experiment 03 must have passed the gate (MAD < 0.05).

Outputs:
    results/plots/04_hedge_ratio_with_costs.png

Prints to stdout:
    CVaR and expected cost comparison (neural vs classical) at 5 bps.
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
from src.train import train

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
COST_RATE = 0.0005          # 5 basis points
N_TRAIN_PATHS = 20_000
N_TRAIN_EPOCHS = 300
TRAIN_SEED = 43             # different seed from experiment 03
EVAL_SEED = TRAIN_SEED + 10_000
N_EVAL_PATHS = 20_000
ALPHA = 0.95
PLOT_PATH = "results/plots/04_hedge_ratio_with_costs.png"


def collect_deltas(
    model: HedgeNet,
    paths: np.ndarray,
    market: MarketConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (model_deltas, bs_deltas) at every (path, step) state.

    Shape: (n_paths * n_steps,) each.
    """
    n_paths, n_steps_plus_1 = paths.shape
    n_steps = n_steps_plus_1 - 1
    dt = market.T / n_steps

    paths_t = torch.tensor(paths, dtype=torch.float32)
    model_out: list[np.ndarray] = []
    bs_out: list[np.ndarray] = []

    delta_prev = torch.zeros(n_paths, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        for i in range(n_steps):
            s_i = paths_t[:, i]
            tau_i = market.T - i * dt
            feats = torch.stack(
                [s_i / market.s0, torch.full((n_paths,), tau_i), delta_prev],
                dim=1,
            )
            delta_model = model(feats)
            delta_bs = np.asarray(
                bs.call_delta(paths[:, i], market.k, market.r, market.sigma, tau_i)
            )
            model_out.append(delta_model.numpy())
            bs_out.append(delta_bs)
            delta_prev = delta_model

    return np.concatenate(model_out), np.concatenate(bs_out)


def main() -> None:
    market = MarketConfig()
    cfg = TrainingConfig(
        n_paths=N_TRAIN_PATHS,
        n_epochs=N_TRAIN_EPOCHS,
        seed=TRAIN_SEED,
        cost_rate=COST_RATE,
        alpha=ALPHA,
    )
    bps = int(round(COST_RATE * 10_000))

    print("=" * 60)
    print("Experiment 04 — Neural Hedger: With Transaction Costs")
    print(f"  cost_rate={COST_RATE} ({bps} bps)   n_epochs={N_TRAIN_EPOCHS}")
    print("-" * 60)

    model = HedgeNet(n_layers=4, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model, loss_history = train(cfg, market, model, optimizer)
    print(f"  Training complete. Initial loss={loss_history[0]:.4f}  Final loss={loss_history[-1]:.4f}")

    # ---- Evaluate on held-out paths ----
    gbm_rn = GBM(s0=market.s0, mu=market.r, sigma=market.sigma)
    eval_paths = gbm_rn.sample_paths(
        n_paths=N_EVAL_PATHS, n_steps=market.n_steps, T=market.T, seed=EVAL_SEED
    )
    tgrid = time_grid(n_steps=market.n_steps, T=market.T)
    initial_price = float(bs.call_price(market.s0, market.k, market.r, market.sigma, market.T))

    # Neural P&L at this cost level
    from src.train import _compute_pnl
    with torch.no_grad():
        model.eval()
        paths_t = torch.tensor(eval_paths, dtype=torch.float32)
        pnl_neural = _compute_pnl(
            paths=paths_t,
            model=model,
            s0=market.s0,
            k=market.k,
            r=market.r,
            T=market.T,
            n_steps=market.n_steps,
            cost_rate=COST_RATE,
            initial_option_price=initial_price,
        ).numpy()

    # Classical hedger P&L
    pnl_classical = classical_delta_pnl(
        paths=eval_paths,
        time_grid=tgrid,
        k=market.k,
        r=market.r,
        sigma=market.sigma,
        cost_rate=COST_RATE,
        initial_option_price=initial_price,
    )

    cvar_neural = cvar(pnl_neural, alpha=ALPHA)
    cvar_classical = cvar(pnl_classical, alpha=ALPHA)
    # Expected cost = mean P&L reduction due to costs (vs no-cost classical)
    pnl_classical_nc = classical_delta_pnl(
        paths=eval_paths, time_grid=tgrid, k=market.k, r=market.r, sigma=market.sigma,
        cost_rate=0.0, initial_option_price=initial_price,
    )
    expected_cost_classical = float(pnl_classical_nc.mean() - pnl_classical.mean())

    # For neural: compare to classical no-cost to get total cost paid
    expected_cost_neural = float(pnl_classical_nc.mean() - pnl_neural.mean())

    # Delta comparison
    model_deltas, bs_deltas = collect_deltas(model, eval_paths, market)
    mad_from_bs = float(np.abs(model_deltas - bs_deltas).mean())

    print(f"  Classical  CVaR={cvar_classical:.4f}   E[cost]={expected_cost_classical:.4f}")
    print(f"  Neural     CVaR={cvar_neural:.4f}   E[cost]={expected_cost_neural:.4f}")
    print(f"  Neural MAD from BS delta: {mad_from_bs:.4f} (should be > exp 03 MAD — network is adapting)")
    print(f"  Saved plot to: {PLOT_PATH}")
    print("=" * 60)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: delta scatter at this cost level (neural vs BS)
    ax = axes[0]
    rng = np.random.default_rng(0)
    n_plot = min(8_000, len(model_deltas))
    idx = rng.choice(len(model_deltas), size=n_plot, replace=False)
    ax.scatter(
        bs_deltas[idx], model_deltas[idx],
        s=2, alpha=0.25, color="darkorange",
        label=f"Neural delta at {bps} bps",
    )
    ax.plot([0, 1], [0, 1], color="steelblue", linewidth=1.5, label="Identity (BS delta)")
    ax.set_xlabel("Black-Scholes delta", fontsize=12)
    ax.set_ylabel("Neural hedger delta", fontsize=12)
    ax.set_title(f"Neural vs BS Delta  ({bps} bps cost)", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # Right: P&L distribution comparison
    ax2 = axes[1]
    lo = min(pnl_neural.min(), pnl_classical.min())
    hi = max(pnl_neural.max(), pnl_classical.max())
    bins = np.linspace(lo, hi, 80)
    ax2.hist(pnl_classical, bins=bins, alpha=0.6, color="steelblue",
             label=f"Classical  CVaR={cvar_classical:.3f}")
    ax2.hist(pnl_neural, bins=bins, alpha=0.6, color="darkorange",
             label=f"Neural     CVaR={cvar_neural:.3f}")
    ax2.axvline(pnl_classical.mean(), color="steelblue", linestyle="--", linewidth=1.3)
    ax2.axvline(pnl_neural.mean(), color="darkorange", linestyle="--", linewidth=1.3)
    ax2.set_xlabel("Terminal P&L", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title(f"P&L Distribution at {bps} bps", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        f"Neural Hedger Under Transaction Costs ({bps} bps)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()

    os.makedirs("plots", exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
