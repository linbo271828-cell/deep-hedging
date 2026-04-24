"""Experiment 03: Neural hedger sanity check — zero transaction costs.

Trains the neural hedger with zero transaction costs and verifies that the
learned hedge ratio closely approximates the Black-Scholes delta.  This is
the gate experiment: if the network cannot recover BS delta in the frictionless
case, we cannot trust its behaviour under transaction costs.

Gate criterion: mean absolute deviation (MAD) between learned and BS delta
must be < 0.05 across all (path, step) states in the evaluation set.

Outputs:
    results/plots/03_learned_vs_bs_delta.png   (scatter + loss curve)

Prints to stdout:
    Training loss progression, final MAD, pass/fail verdict.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import src.black_scholes as bs
from src.config import MarketConfig, TrainingConfig
from src.market import GBM, time_grid
from src.neural_hedger import HedgeNet
from src.train import train

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
TRAIN_SEED = 42
EVAL_SEED = TRAIN_SEED + 10_000   # independent evaluation paths
N_TRAIN_PATHS = 20_000
N_TRAIN_EPOCHS = 300
N_EVAL_PATHS = 20_000
MAD_GATE = 0.05                   # maximum acceptable mean absolute deviation
PLOT_PATH = "results/plots/03_learned_vs_bs_delta.png"


def collect_deltas(
    model: HedgeNet,
    paths: np.ndarray,
    market: MarketConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (model_deltas, bs_deltas) at every (path, step) state.

    Shape of each output: (n_paths * n_steps,) — flattened across paths and steps.
    Only interior steps (i=0..n_steps-1) are included; the terminal step has
    tau=0 so delta is determined by the payoff, not the hedging policy.
    """
    n_paths, n_steps_plus_1 = paths.shape
    n_steps = n_steps_plus_1 - 1
    dt = market.T / n_steps

    paths_t = torch.tensor(paths, dtype=torch.float32)
    model_deltas: list[np.ndarray] = []
    bs_deltas_list: list[np.ndarray] = []

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

            model_deltas.append(delta_model.numpy())
            bs_deltas_list.append(delta_bs)
            delta_prev = delta_model

    return (
        np.concatenate(model_deltas),
        np.concatenate(bs_deltas_list),
    )


def main() -> None:
    market = MarketConfig()
    cfg = TrainingConfig(
        n_paths=N_TRAIN_PATHS,
        n_epochs=N_TRAIN_EPOCHS,
        seed=TRAIN_SEED,
        cost_rate=0.0,
        alpha=0.95,
    )

    model = HedgeNet(n_layers=4, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print("=" * 60)
    print("Experiment 03 — Neural Hedger: Zero Transaction Costs")
    print(f"  n_train_paths={N_TRAIN_PATHS:,}   n_epochs={N_TRAIN_EPOCHS}"
          f"   cost_rate={cfg.cost_rate}")
    print("-" * 60)

    model, loss_history = train(cfg, market, model, optimizer)

    # Print loss at 0%, 25%, 50%, 75%, 100% of training
    checkpoints = [0, N_TRAIN_EPOCHS // 5, 2 * N_TRAIN_EPOCHS // 5,
                   3 * N_TRAIN_EPOCHS // 5, 4 * N_TRAIN_EPOCHS // 5, N_TRAIN_EPOCHS - 1]
    for ep in checkpoints:
        print(f"  Epoch {ep+1:>3d}  CVaR loss = {loss_history[ep]:.4f}")

    # ---- Evaluate on held-out paths ----
    gbm_rn = GBM(s0=market.s0, mu=market.r, sigma=market.sigma)
    eval_paths = gbm_rn.sample_paths(
        n_paths=N_EVAL_PATHS, n_steps=market.n_steps, T=market.T, seed=EVAL_SEED
    )

    model_deltas, bs_deltas = collect_deltas(model, eval_paths, market)

    mad = float(np.abs(model_deltas - bs_deltas).mean())
    gate_pass = mad < MAD_GATE

    print("-" * 60)
    print(f"  MAD (model vs BS delta) = {mad:.4f}   gate threshold = {MAD_GATE}")
    print(f"  Gate status: {'PASS ✓' if gate_pass else 'FAIL ✗ — do NOT proceed to exp 04/05'}")
    print(f"  Saved plot to: {PLOT_PATH}")
    print("=" * 60)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: training loss curve
    ax = axes[0]
    ax.plot(range(1, len(loss_history) + 1), loss_history, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("CVaR Loss", fontsize=12)
    ax.set_title("Training Loss (Zero Costs)", fontsize=13)
    ax.grid(alpha=0.3)

    # Right: scatter of model delta vs BS delta (subsample to avoid overplotting)
    ax2 = axes[1]
    rng = np.random.default_rng(0)
    n_plot = min(10_000, len(model_deltas))
    idx = rng.choice(len(model_deltas), size=n_plot, replace=False)
    ax2.scatter(
        bs_deltas[idx],
        model_deltas[idx],
        s=2,
        alpha=0.3,
        color="steelblue",
        label=f"n={n_plot:,} states",
    )
    ax2.plot([0, 1], [0, 1], color="firebrick", linewidth=1.5, label="Identity (perfect)")
    ax2.set_xlabel("Black-Scholes delta", fontsize=12)
    ax2.set_ylabel("Neural hedger delta", fontsize=12)
    ax2.set_title(f"Learned vs BS Delta  (MAD = {mad:.4f})", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "Neural Hedger Sanity Check — Zero Transaction Costs",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()

    os.makedirs("plots", exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
