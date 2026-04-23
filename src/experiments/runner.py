"""Experiment runner — thin orchestrator for end-to-end experiment execution.

Lane E (experiment system) owns this file.
No math lives here. All numerical logic delegates to src/neural/, src/hedging/, src/risk/.

Artifact layout (per experiment):
  results/raw/<name>/
    config.json              ExperimentConfig serialized
    neural_summary.json      SummaryRow for neural strategy
    classical_summary.json   SummaryRow for classical strategy
    loss_history.json        per-epoch CVaR during training
    model_checkpoint.pt      trained model weights
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.experiments.registry import get_config
from src.experiments.summaries import SummaryRow
from src.hedging.baseline import classical_delta_pnl
from src.hedging.transaction_costs import proportional_cost
from src.models.gbm import GBM, time_grid as make_time_grid
from src.neural.architectures import HedgeNet
from src.neural.evaluate import evaluate
from src.neural.train import train
from src.payoffs import european as bs
from src.payoffs.european import call_price
from src.risk.cvar import cvar
from src.risk.entropic import entropic_risk
from src.utils.io import load_json, save_json
from src.utils.seeds import derive_seed

if TYPE_CHECKING:
    from src.experiments.configs import ExperimentConfig


def _results_dir(name: str) -> str:
    return os.path.join("results", "raw", name)


def _already_done(out_dir: str) -> bool:
    return (
        os.path.exists(os.path.join(out_dir, "neural_summary.json"))
        and os.path.exists(os.path.join(out_dir, "classical_summary.json"))
    )


def _eval_classical(config: ExperimentConfig) -> SummaryRow:
    """Evaluate the classical BS delta hedge on held-out evaluation paths."""
    mc = config.market
    tc = config.training
    ec = config.evaluation

    eval_seed = derive_seed(tc.seed + ec.seed_offset, "eval")
    gbm = GBM(s0=mc.s0, mu=mc.r, sigma=mc.sigma)
    paths = gbm.sample_paths(
        n_paths=ec.n_paths,
        n_steps=mc.n_steps,
        T=mc.T,
        seed=eval_seed,
    )
    tg = make_time_grid(mc.n_steps, mc.T)

    initial_option_price = float(call_price(mc.s0, mc.k, mc.r, mc.sigma, mc.T))
    pnl = classical_delta_pnl(
        paths=paths,
        time_grid=tg,
        k=mc.k,
        r=mc.r,
        sigma=mc.sigma,
        cost_rate=tc.cost_rate,
        initial_option_price=initial_option_price,
    )

    n_paths_count = paths.shape[0]
    n_steps = paths.shape[1] - 1
    T = tg[-1]

    total_cost = np.zeros(n_paths_count)
    total_turnover = np.zeros(n_paths_count)

    delta = bs.call_delta(paths[:, 0], mc.k, mc.r, mc.sigma, T)
    cost0 = proportional_cost(delta, paths[:, 0], tc.cost_rate)
    total_cost += cost0
    total_turnover += np.abs(delta)

    for i in range(1, n_steps):
        tau_i = T - tg[i]
        delta_new = bs.call_delta(paths[:, i], mc.k, mc.r, mc.sigma, tau_i)
        trade = delta_new - delta
        total_cost += proportional_cost(trade, paths[:, i], tc.cost_rate)
        total_turnover += np.abs(trade)
        delta = delta_new

    liquidation_cost = proportional_cost(delta, paths[:, n_steps], tc.cost_rate)
    total_cost += liquidation_cost
    total_turnover += np.abs(delta)

    return SummaryRow(
        experiment_name=config.name,
        strategy="classical",
        seed=tc.seed,
        cost_rate=tc.cost_rate,
        mean_pnl=float(np.mean(pnl)),
        std_pnl=float(np.std(pnl)),
        cvar_95=float(cvar(pnl, alpha=ec.alpha)),
        entropic_risk=float(entropic_risk(pnl, lambda_=ec.lambda_)),
        expected_cost=float(np.mean(total_cost)),
        turnover=float(np.mean(total_turnover)),
        n_paths=ec.n_paths,
        market_model="gbm",
        payoff_type=config.payoff.payoff_type,
        hedge_universe=config.hedge_universe.universe_type,
        n_epochs=None,
        hidden_dim=None,
        n_layers=None,
    )


def _run_core(config: ExperimentConfig, out_dir: str) -> list[SummaryRow]:
    """Execute training + evaluation and write all artifacts to out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    config_dict = dataclasses.asdict(config)
    save_json(config_dict, os.path.join(out_dir, "config.json"))

    tc = config.training
    mc = config.market
    ec = config.evaluation

    model = HedgeNet(n_layers=tc.n_layers, hidden_dim=tc.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc.lr)
    model, loss_history = train(config=tc, market_config=mc, model=model, optimizer=optimizer)

    torch.save(model.state_dict(), os.path.join(out_dir, "model_checkpoint.pt"))
    save_json({"loss_history": loss_history}, os.path.join(out_dir, "loss_history.json"))

    eval_result = evaluate(model=model, market_config=mc, training_config=tc, eval_config=ec)

    neural_row = SummaryRow(
        experiment_name=config.name,
        strategy="neural",
        seed=tc.seed,
        cost_rate=tc.cost_rate,
        mean_pnl=eval_result.mean_pnl,
        std_pnl=eval_result.std_pnl,
        cvar_95=eval_result.cvar_95,
        entropic_risk=eval_result.entropic_risk,
        expected_cost=eval_result.expected_cost,
        turnover=eval_result.turnover,
        n_paths=eval_result.n_paths,
        market_model="gbm",
        payoff_type=config.payoff.payoff_type,
        hedge_universe=config.hedge_universe.universe_type,
        n_epochs=tc.n_epochs,
        hidden_dim=tc.hidden_dim,
        n_layers=tc.n_layers,
    )

    classical_row = _eval_classical(config)

    save_json(neural_row.to_dict(), os.path.join(out_dir, "neural_summary.json"))
    save_json(classical_row.to_dict(), os.path.join(out_dir, "classical_summary.json"))

    return [neural_row, classical_row]


def run_experiment(name: str, force: bool = False) -> list[SummaryRow]:
    """Run a named experiment end-to-end and return summary rows.

    Steps:
      1. Resolve ExperimentConfig from registry.
      2. Skip if artifacts already exist and force=False.
      3. Train neural hedger.
      4. Evaluate neural hedger on held-out paths.
      5. Evaluate classical BS delta hedge on the same held-out paths.
      6. Save all artifacts to results/raw/<name>/.
      7. Return [neural_row, classical_row].

    Parameters
    ----------
    name:
        Experiment name from the registry.
    force:
        Re-run even if results already exist. Default False.

    Returns
    -------
    list[SummaryRow]
        One SummaryRow per strategy: [neural_row, classical_row].
    """
    config = get_config(name)
    out_dir = _results_dir(name)

    if _already_done(out_dir) and not force:
        neural_row = SummaryRow(**load_json(os.path.join(out_dir, "neural_summary.json")))
        classical_row = SummaryRow(**load_json(os.path.join(out_dir, "classical_summary.json")))
        return [neural_row, classical_row]

    return _run_core(config, out_dir)


def run_with_config(config: ExperimentConfig, out_dir: str) -> list[SummaryRow]:
    """Run an experiment given a direct ExperimentConfig (no registry lookup).

    Unlike run_experiment(), the caller provides both the config and output directory.
    Always re-runs — does not check for existing artifacts.

    Parameters
    ----------
    config:
        Fully-specified ExperimentConfig. config.name is used for artifact metadata.
    out_dir:
        Directory where all artifacts will be written. Created if absent.

    Returns
    -------
    list[SummaryRow]
        One SummaryRow per strategy: [neural_row, classical_row].
    """
    return _run_core(config, out_dir)
