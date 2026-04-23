"""Neural hedger evaluation on held-out paths.

Lane D (neural) owns this file.
Separation from training is mandatory: training loss != evaluation metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.models.gbm import GBM
from src.payoffs.european import call_price
from src.risk.cvar import cvar
from src.risk.entropic import entropic_risk as entropic_risk_fn
from src.utils.seeds import derive_seed

if TYPE_CHECKING:
    from src.experiments.configs import EvaluationConfig, MarketConfig, TrainingConfig
    from src.neural.architectures import HedgeNet


@dataclass
class EvaluationResult:
    """Standard metrics for a single hedging strategy evaluation.

    All fields are floats except n_paths (int).
    Larger risk values (cvar_95, entropic_risk) = worse.
    """

    mean_pnl: float
    std_pnl: float
    cvar_95: float
    entropic_risk: float
    expected_cost: float
    turnover: float
    n_paths: int


def _compute_pnl_numpy(
    model: HedgeNet,
    paths: np.ndarray,
    s0: float,
    k: float,
    r: float,
    T: float,
    n_steps: int,
    cost_rate: float,
    initial_option_price: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model inference on paths and return (pnl, total_cost, turnover) arrays.

    Mirrors the accounting in src/hedging/pnl.hedge_pnl and src/neural/train._compute_pnl.
    Uses NumPy + torch.no_grad inference — no gradient tracking.
    """
    n_paths_count = paths.shape[0]
    dt = T / n_steps
    compound = math.exp(r * dt)

    model.eval()
    total_cost = np.zeros(n_paths_count)
    total_turnover = np.zeros(n_paths_count)

    with torch.no_grad():
        paths_t = torch.tensor(paths, dtype=torch.float32)

        s_curr = paths_t[:, 0]
        tau_curr = T

        delta_prev = torch.zeros(n_paths_count, dtype=torch.float32)
        feats = torch.stack(
            [s_curr / s0, torch.full((n_paths_count,), tau_curr), delta_prev],
            dim=1,
        )
        delta = model(feats)

        cost0 = (delta * s_curr * cost_rate).numpy()
        total_cost += cost0
        total_turnover += delta.numpy()  # initial purchase counts as turnover

        cash = initial_option_price - (delta * s_curr).numpy() - cost0
        cash = cash * compound
        delta_np = delta.numpy()

        for i in range(1, n_steps):
            s_curr_np = paths[:, i]
            tau_curr = T - i * dt
            s_curr_t = paths_t[:, i]

            feats = torch.stack(
                [
                    s_curr_t / s0,
                    torch.full((n_paths_count,), tau_curr),
                    torch.tensor(delta_np, dtype=torch.float32),
                ],
                dim=1,
            )
            delta_new = model(feats).numpy()

            trade = delta_new - delta_np
            cost_i = np.abs(trade) * s_curr_np * cost_rate
            total_cost += cost_i
            total_turnover += np.abs(trade)

            cash = cash - trade * s_curr_np - cost_i
            delta_np = delta_new
            cash = cash * compound

        s_T = paths[:, n_steps]
        liquidation_cost = np.abs(delta_np) * s_T * cost_rate
        total_cost += liquidation_cost
        total_turnover += np.abs(delta_np)

        cash = cash + delta_np * s_T - liquidation_cost
        payoff = np.maximum(s_T - k, 0.0)
        pnl = cash - payoff

    return pnl, total_cost, total_turnover


def evaluate_raw(
    model: HedgeNet,
    market_config: MarketConfig,
    training_config: TrainingConfig,
    eval_config: EvaluationConfig,
) -> tuple[EvaluationResult, np.ndarray]:
    """Like evaluate(), but also returns the raw terminal P&L array.

    Used by downstream consumers (e.g. web backend) to generate P&L distribution plots
    without re-running evaluation.

    Returns
    -------
    tuple[EvaluationResult, np.ndarray]
        (metrics, pnl_array) where pnl_array has shape (eval_config.n_paths,).
    """
    eval_seed = derive_seed(training_config.seed + eval_config.seed_offset, "eval")
    gbm = GBM(s0=market_config.s0, mu=market_config.r, sigma=market_config.sigma)
    paths = gbm.sample_paths(
        n_paths=eval_config.n_paths,
        n_steps=market_config.n_steps,
        T=market_config.T,
        seed=eval_seed,
    )

    initial_option_price = float(
        call_price(
            market_config.s0,
            market_config.k,
            market_config.r,
            market_config.sigma,
            market_config.T,
        )
    )

    pnl, total_cost, total_turnover = _compute_pnl_numpy(
        model=model,
        paths=paths,
        s0=market_config.s0,
        k=market_config.k,
        r=market_config.r,
        T=market_config.T,
        n_steps=market_config.n_steps,
        cost_rate=training_config.cost_rate,
        initial_option_price=initial_option_price,
    )

    result = EvaluationResult(
        mean_pnl=float(np.mean(pnl)),
        std_pnl=float(np.std(pnl)),
        cvar_95=float(cvar(pnl, alpha=eval_config.alpha)),
        entropic_risk=float(entropic_risk_fn(pnl, lambda_=eval_config.lambda_)),
        expected_cost=float(np.mean(total_cost)),
        turnover=float(np.mean(total_turnover)),
        n_paths=eval_config.n_paths,
    )
    return result, pnl


def evaluate(
    model: HedgeNet,
    market_config: MarketConfig,
    training_config: TrainingConfig,
    eval_config: EvaluationConfig,
) -> EvaluationResult:
    """Evaluate a trained neural hedger on fresh held-out paths.

    Paths are generated with seed = derive_seed(training.seed + eval.seed_offset, "eval"),
    ensuring independence from training paths across all experiments.

    Parameters
    ----------
    model:
        Trained HedgeNet. Set to eval mode internally; weights are not modified.
    market_config:
        Market parameters (same as used in training).
    training_config:
        Training config (used to read cost_rate, seed).
    eval_config:
        Evaluation settings (n_paths, seed_offset, alpha, lambda_).

    Returns
    -------
    EvaluationResult
        All standard metrics computed on eval_config.n_paths held-out paths.
    """
    result, _ = evaluate_raw(model, market_config, training_config, eval_config)
    return result
