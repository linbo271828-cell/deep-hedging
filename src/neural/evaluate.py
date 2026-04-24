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
    from src.experiments.configs import EvaluationConfig, HedgeUniverseConfig, MarketConfig, PayoffConfig, TrainingConfig
    from src.neural.architectures import HedgeNet, HedgeNetMulti


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
    model: "HedgeNet",
    paths: np.ndarray,
    s0: float,
    k: float,
    r: float,
    T: float,
    n_steps: int,
    cost_rate: float,
    initial_option_price: float,
    payoff_fn: object = None,
    delta_transform: object = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model inference on paths and return (pnl, total_cost, turnover) arrays.

    Mirrors the accounting in src/hedging/pnl.hedge_pnl and src/neural/train._compute_pnl.
    Uses NumPy + torch.no_grad inference — no gradient tracking.

    payoff_fn: callable(s_T np.ndarray) -> np.ndarray.  None = European call.
    delta_transform: callable(model_output np.ndarray) -> np.ndarray.  None = identity.
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
        raw = model(feats)
        delta = delta_transform(raw) if delta_transform is not None else raw

        cost0 = (torch.abs(delta) * s_curr * cost_rate).numpy()
        total_cost += cost0
        total_turnover += np.abs(delta.numpy())

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
            raw_new = model(feats)
            delta_new = (
                delta_transform(raw_new).numpy()
                if delta_transform is not None
                else raw_new.numpy()
            )

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
        payoff = payoff_fn(s_T) if payoff_fn is not None else np.maximum(s_T - k, 0.0)
        pnl = cash - payoff

    return pnl, total_cost, total_turnover


def evaluate_raw(
    model: "HedgeNet",
    market_config: "MarketConfig",
    training_config: "TrainingConfig",
    eval_config: "EvaluationConfig",
    payoff_config: "PayoffConfig | None" = None,
) -> tuple[EvaluationResult, np.ndarray]:
    """Like evaluate(), but also returns the raw terminal P&L array.

    Used by downstream consumers (e.g. web backend) to generate P&L distribution plots
    without re-running evaluation.

    Parameters
    ----------
    payoff_config:
        If provided (and not "call"), uses the generalized payoff dispatch.
        None defaults to European call for backward compatibility.

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

    if payoff_config is not None and payoff_config.payoff_type != "call":
        from src.payoffs.dispatch import get_payoff_spec
        spec = get_payoff_spec(payoff_config, market_config)
        initial_option_price = spec.initial_option_price
        payoff_fn = spec.payoff_fn
        delta_transform = spec.delta_transform
    else:
        initial_option_price = float(
            call_price(
                market_config.s0,
                market_config.k,
                market_config.r,
                market_config.sigma,
                market_config.T,
            )
        )
        payoff_fn = None
        delta_transform = None

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
        payoff_fn=payoff_fn,
        delta_transform=delta_transform,
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
    model: "HedgeNet",
    market_config: "MarketConfig",
    training_config: "TrainingConfig",
    eval_config: "EvaluationConfig",
    payoff_config: "PayoffConfig | None" = None,
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
    result, _ = evaluate_raw(model, market_config, training_config, eval_config, payoff_config)
    return result


# ---------------------------------------------------------------------------
# Multi-instrument evaluation (stock + option hedge universe)
# ---------------------------------------------------------------------------

def evaluate_raw_multi(
    model: "HedgeNetMulti",
    market_config: "MarketConfig",
    training_config: "TrainingConfig",
    eval_config: "EvaluationConfig",
    hedge_universe_config: "HedgeUniverseConfig",
    payoff_config: "PayoffConfig | None" = None,
) -> tuple["EvaluationResult", np.ndarray]:
    """Evaluate a HedgeNetMulti on held-out paths for the stock + option universe.

    Returns
    -------
    tuple[EvaluationResult, np.ndarray]
        (metrics, pnl_array) where pnl_array has shape (eval_config.n_paths,).
    """
    from src.hedging.hedge_universe import StockPlusOptionUniverse
    from src.models.gbm import time_grid as make_time_grid

    eval_seed = derive_seed(training_config.seed + eval_config.seed_offset, "eval")
    gbm = GBM(s0=market_config.s0, mu=market_config.r, sigma=market_config.sigma)
    paths = gbm.sample_paths(
        n_paths=eval_config.n_paths,
        n_steps=market_config.n_steps,
        T=market_config.T,
        seed=eval_seed,
    )
    tg = make_time_grid(market_config.n_steps, market_config.T)

    universe = StockPlusOptionUniverse(market_config, hedge_universe_config)
    hedge_prices = universe.hedge_option_prices_along_paths(paths, tg)

    k_h = hedge_universe_config.hedge_option_strike
    v_hedge_0 = float(universe.hedge_option_price(
        np.array([market_config.s0]), market_config.T
    )[0])
    if v_hedge_0 < 1e-8:
        v_hedge_0 = 1.0

    if payoff_config is not None and payoff_config.payoff_type != "call":
        from src.payoffs.dispatch import get_payoff_spec
        spec = get_payoff_spec(payoff_config, market_config)
        initial_option_price = spec.initial_option_price
        payoff_fn = spec.payoff_fn
        delta_transform = spec.delta_transform
    else:
        initial_option_price = float(
            call_price(market_config.s0, market_config.k, market_config.r,
                       market_config.sigma, market_config.T)
        )
        payoff_fn = None
        delta_transform = None

    n_paths_count = paths.shape[0]
    n_steps = paths.shape[1] - 1
    T = tg[-1]
    dt = T / n_steps

    model.eval()
    total_cost = np.zeros(n_paths_count)
    total_turnover = np.zeros(n_paths_count)

    with torch.no_grad():
        paths_t = torch.tensor(paths, dtype=torch.float32)
        hedge_t = torch.tensor(hedge_prices, dtype=torch.float32)

        s0 = market_config.s0
        ds_prev = torch.zeros(n_paths_count, dtype=torch.float32)
        dh_prev = torch.zeros(n_paths_count, dtype=torch.float32)

        s_curr = paths_t[:, 0]
        h_curr = hedge_t[:, 0]

        feats = torch.stack(
            [s_curr / s0,
             torch.full((n_paths_count,), T),
             ds_prev, dh_prev,
             h_curr / v_hedge_0],
            dim=1,
        )
        out = model(feats)
        ds = (delta_transform(out[:, 0]) if delta_transform is not None else out[:, 0])
        dh = out[:, 1]

        cost0_s = (torch.abs(ds) * s_curr * training_config.cost_rate).numpy()
        cost0_h = (torch.abs(dh) * h_curr * training_config.cost_rate).numpy()
        total_cost += cost0_s + cost0_h
        total_turnover += np.abs(ds.numpy()) + np.abs(dh.numpy())

        cash = (initial_option_price
                - (ds * s_curr).numpy() - cost0_s
                - (dh * h_curr).numpy() - cost0_h)
        ds_np = ds.numpy()
        dh_np = dh.numpy()

        for i in range(1, n_steps):
            s_curr_np = paths[:, i]
            h_curr_np = hedge_prices[:, i]
            tau_i = T - i * dt
            s_curr_t = paths_t[:, i]
            h_curr_t = hedge_t[:, i]

            feats = torch.stack(
                [s_curr_t / s0,
                 torch.full((n_paths_count,), tau_i),
                 torch.tensor(ds_np, dtype=torch.float32),
                 torch.tensor(dh_np, dtype=torch.float32),
                 h_curr_t / v_hedge_0],
                dim=1,
            )
            out_new = model(feats)
            ds_new = (
                delta_transform(out_new[:, 0]).numpy()
                if delta_transform is not None
                else out_new[:, 0].numpy()
            )
            dh_new = out_new[:, 1].numpy()

            trade_s = ds_new - ds_np
            trade_h = dh_new - dh_np
            cost_s = np.abs(trade_s) * s_curr_np * training_config.cost_rate
            cost_h = np.abs(trade_h) * h_curr_np * training_config.cost_rate
            total_cost += cost_s + cost_h
            total_turnover += np.abs(trade_s) + np.abs(trade_h)

            cash = cash - trade_s * s_curr_np - cost_s - trade_h * h_curr_np - cost_h
            ds_np = ds_new
            dh_np = dh_new

        s_T = paths[:, n_steps]
        liq_cost_s = np.abs(ds_np) * s_T * training_config.cost_rate
        total_cost += liq_cost_s
        total_turnover += np.abs(ds_np)

        cash = cash + ds_np * s_T - liq_cost_s

        hedge_payoff = np.maximum(s_T - k_h, 0.0)
        cash = cash + dh_np * hedge_payoff

        payoff = payoff_fn(s_T) if payoff_fn is not None else np.maximum(s_T - 0.0, 0.0)
        pnl = cash - payoff

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
