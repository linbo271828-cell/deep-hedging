"""FastAPI backend for Deep Hedging Lab.

Serves preset metadata, launches experiment runs, polls run status,
and returns metrics + P&L distribution plots.

Execution model: runs execute synchronously in a background thread per request.
In-memory run registry (RUNS dict) is sufficient for single-machine dev use.

Start with: PYTHONPATH=. uvicorn services.sim.main:app --reload --port 8000
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure repo root is on sys.path when run from any working directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.experiments.configs import (
    EvaluationConfig,
    ExperimentConfig,
    HedgeUniverseConfig,
    MarketConfig,
    PayoffConfig,
    TrainingConfig,
)
from src.experiments.runner import run_with_config
from src.hedging.baseline import classical_delta_pnl_generalized
from src.models.gbm import GBM, time_grid as make_time_grid
from src.neural.architectures import HedgeNet
from src.neural.evaluate import evaluate_raw
from src.payoffs.dispatch import get_payoff_spec
from src.payoffs.european import call_delta, call_price
from src.utils.seeds import derive_seed

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Deep Hedging Lab API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Payoff metadata — single source of truth for UI-facing descriptions
# ---------------------------------------------------------------------------

_PAYOFF_METADATA: dict[str, dict[str, str]] = {
    "call": {
        "payoff_display_name": "European Call",
        "payoff_formula": "max(S_T − K, 0)",
        "payoff_profile": "Unbounded upside; zero payoff when the stock finishes below strike.",
        "delta_range": "[0, 1]",
        "benchmark_label": "BS Call Δ = N(d₁)",
        "classical_description": "Black-Scholes call delta N(d₁)",
    },
    "put": {
        "payoff_display_name": "European Put",
        "payoff_formula": "max(K − S_T, 0)",
        "payoff_profile": "Profits from downward moves; zero payoff above strike. "
                          "Negative delta — the hedger holds a short stock position.",
        "delta_range": "[-1, 0]",
        "benchmark_label": "BS Put Δ = N(d₁) − 1",
        "classical_description": "Black-Scholes put delta N(d₁)−1",
    },
    "call_spread": {
        "payoff_display_name": "Bull Call Spread",
        "payoff_formula": "max(S_T − K_lo, 0) − max(S_T − K_hi, 0)",
        "payoff_profile": "Bounded payoff capped at K_hi − K_lo. "
                          "No single analytic BS delta — benchmark uses net leg deltas.",
        "delta_range": "[0, 1]",
        "benchmark_label": "Net BS Δ = Δ_call(K_lo) − Δ_call(K_hi)",
        "classical_description": "Net BS delta: call_delta(K_lo)−call_delta(K_hi)",
    },
    "straddle": {
        "payoff_display_name": "Straddle",
        "payoff_formula": "max(S_T − K, 0) + max(K − S_T, 0) = |S_T − K|",
        "payoff_profile": "Profits from large moves in either direction. "
                          "ATM delta ≈ 0 by symmetry; the network learns a near-neutral position.",
        "delta_range": "[-1, 1]",
        "benchmark_label": "Net BS Δ = 2·N(d₁) − 1",
        "classical_description": "Net BS delta: call_delta+put_delta = 2·N(d₁)−1",
    },
}


def _payoff_meta(payoff_type_key: str) -> dict[str, str]:
    """Look up payoff metadata by internal key (call / put / call_spread / straddle)."""
    return _PAYOFF_METADATA.get(payoff_type_key, {})


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

_FAST_EVAL = EvaluationConfig(n_paths=20_000, seed_offset=1000)

PRESETS: dict[str, dict[str, Any]] = {
    "gbm_call_zero_cost": {
        "id": "gbm_call_zero_cost",
        "name": "GBM Call — Zero Cost",
        "description": (
            "The zero-cost baseline: neural hedger trained with no transaction costs. "
            "At convergence the network must recover the Black-Scholes delta, providing "
            "a clean theoretical gate. Pass this and the network is learning something real."
        ),
        **_payoff_meta("call"),
        "cost_rate": 0.0,
        "market_model": "GBM",
        "payoff_type": "European Call",
        "hedge_universe": "Stock Only",
        "n_epochs": 40,
        "est_seconds": 35,
        "config": ExperimentConfig(
            name="gbm_call_zero_cost",
            market=MarketConfig(),
            payoff=PayoffConfig(payoff_type="call"),
            hedge_universe=HedgeUniverseConfig(universe_type="stock_only"),
            training=TrainingConfig(
                n_paths=8_000, n_epochs=40, hidden_dim=32, n_layers=3,
                seed=42, cost_rate=0.0,
            ),
            evaluation=_FAST_EVAL,
        ),
    },
    "gbm_call_5bps": {
        "id": "gbm_call_5bps",
        "name": "GBM Call — 5 bps Cost",
        "description": (
            "Neural hedger trained under 5 bp proportional transaction costs. "
            "The optimal strategy is no longer BS delta — the network learns to "
            "trade less aggressively, reducing friction while controlling CVaR. "
            "This is the headline result from Buehler et al. (2019)."
        ),
        **_payoff_meta("call"),
        "cost_rate": 0.0005,
        "market_model": "GBM",
        "payoff_type": "European Call",
        "hedge_universe": "Stock Only",
        "n_epochs": 40,
        "est_seconds": 35,
        "config": ExperimentConfig(
            name="gbm_call_5bps",
            market=MarketConfig(),
            payoff=PayoffConfig(payoff_type="call"),
            hedge_universe=HedgeUniverseConfig(universe_type="stock_only"),
            training=TrainingConfig(
                n_paths=8_000, n_epochs=40, hidden_dim=32, n_layers=3,
                seed=43, cost_rate=0.0005,
            ),
            evaluation=_FAST_EVAL,
        ),
    },
    "gbm_put_5bps": {
        "id": "gbm_put_5bps",
        "name": "GBM Put — 5 bps Cost",
        "description": (
            "Short European put hedged with 5 bp costs. "
            "The classical benchmark is BS put delta N(d₁)−1 ∈ [−1, 0]: "
            "a short stock position that grows toward zero as the stock rises above strike. "
            "Tests whether deep hedging generalises to negatively-sloped payoffs."
        ),
        **_payoff_meta("put"),
        "cost_rate": 0.0005,
        "market_model": "GBM",
        "payoff_type": "European Put",
        "hedge_universe": "Stock Only",
        "n_epochs": 40,
        "est_seconds": 35,
        "config": ExperimentConfig(
            name="gbm_put_5bps",
            market=MarketConfig(),
            payoff=PayoffConfig(payoff_type="put"),
            hedge_universe=HedgeUniverseConfig(universe_type="stock_only"),
            training=TrainingConfig(
                n_paths=8_000, n_epochs=40, hidden_dim=32, n_layers=3,
                seed=44, cost_rate=0.0005,
            ),
            evaluation=_FAST_EVAL,
        ),
    },
    "gbm_call_spread_5bps": {
        "id": "gbm_call_spread_5bps",
        "name": "GBM Bull Call Spread — 5 bps Cost",
        "description": (
            "Short bull call spread (long K=95, short K=105) under 5 bp costs. "
            "The payoff is bounded ∈ [0, 10] — no single BS delta applies. "
            "The classical benchmark sums the two leg deltas. "
            "Bounded payoff changes the risk-reduction dynamic compared to a naked call."
        ),
        **_payoff_meta("call_spread"),
        "cost_rate": 0.0005,
        "market_model": "GBM",
        "payoff_type": "Bull Call Spread",
        "hedge_universe": "Stock Only",
        "n_epochs": 40,
        "est_seconds": 35,
        "config": ExperimentConfig(
            name="gbm_call_spread_5bps",
            market=MarketConfig(),
            payoff=PayoffConfig(payoff_type="call_spread", k=95.0, k_hi=105.0),
            hedge_universe=HedgeUniverseConfig(universe_type="stock_only"),
            training=TrainingConfig(
                n_paths=8_000, n_epochs=40, hidden_dim=32, n_layers=3,
                seed=45, cost_rate=0.0005,
            ),
            evaluation=_FAST_EVAL,
        ),
    },
    "gbm_straddle_5bps": {
        "id": "gbm_straddle_5bps",
        "name": "GBM Straddle — 5 bps Cost",
        "description": (
            "Short straddle (long call + long put at K=100) under 5 bp costs. "
            "Payoff = |S_T − K|; the seller profits from low realised volatility. "
            "The classical benchmark 2·N(d₁)−1 is near zero ATM and ±1 deep ITM/OTM. "
            "The neural hedger must learn to be nearly delta-neutral at-the-money."
        ),
        **_payoff_meta("straddle"),
        "cost_rate": 0.0005,
        "market_model": "GBM",
        "payoff_type": "Straddle",
        "hedge_universe": "Stock Only",
        "n_epochs": 40,
        "est_seconds": 35,
        "config": ExperimentConfig(
            name="gbm_straddle_5bps",
            market=MarketConfig(),
            payoff=PayoffConfig(payoff_type="straddle"),
            hedge_universe=HedgeUniverseConfig(universe_type="stock_only"),
            training=TrainingConfig(
                n_paths=8_000, n_epochs=40, hidden_dim=32, n_layers=3,
                seed=46, cost_rate=0.0005,
            ),
            evaluation=_FAST_EVAL,
        ),
    },
}

_PRESET_LIST = [
    {k: v for k, v in preset.items() if k != "config"}
    for preset in PRESETS.values()
]

# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------

@dataclass
class RunState:
    run_id: str
    preset_id: str
    preset_name: str
    status: str  # "running" | "done" | "error"
    created_at: str
    completed_at: str | None
    error: str | None
    out_dir: str


RUNS: dict[str, RunState] = {}
RUNS_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _generate_pnl_plot(out_dir: str, config: ExperimentConfig) -> str:
    """Generate a P&L distribution histogram comparing neural vs classical.

    Loads the trained model checkpoint, re-runs inference on eval paths,
    re-runs the classical baseline on the same paths, and saves a PNG.
    """
    mc = config.market
    tc = config.training
    ec = config.evaluation

    spec = get_payoff_spec(config.payoff, mc)

    # Load trained model
    model = HedgeNet(n_layers=tc.n_layers, hidden_dim=tc.hidden_dim)
    checkpoint_path = os.path.join(out_dir, "model_checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))

    # Neural P&L
    _, neural_pnl = evaluate_raw(model, mc, tc, ec, payoff_config=config.payoff)

    # Classical P&L (same eval paths via same seed)
    eval_seed = derive_seed(tc.seed + ec.seed_offset, "eval")
    gbm = GBM(s0=mc.s0, mu=mc.r, sigma=mc.sigma)
    paths = gbm.sample_paths(n_paths=ec.n_paths, n_steps=mc.n_steps, T=mc.T, seed=eval_seed)
    tg = make_time_grid(mc.n_steps, mc.T)
    classical_pnl = classical_delta_pnl_generalized(
        paths=paths, time_grid=tg, payoff_spec=spec, cost_rate=tc.cost_rate,
    )

    # Histogram
    all_pnl = np.concatenate([neural_pnl, classical_pnl])
    lo = float(np.percentile(all_pnl, 1))
    hi = float(np.percentile(all_pnl, 99))
    bins = np.linspace(lo, hi, 80)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, pnl, label, color in [
        (axes[0], neural_pnl, "Neural Hedger", "#3B82F6"),
        (axes[1], classical_pnl, "Classical (BS Δ)", "#10B981"),
    ]:
        ax.hist(pnl, bins=bins, color=color, alpha=0.72, density=True,
                edgecolor="white", linewidth=0.3)
        mean_v = float(np.mean(pnl))
        p5_v = float(np.percentile(pnl, 5))
        ax.axvline(mean_v, color="black", linestyle="--", linewidth=1.5,
                   label=f"Mean: {mean_v:.3f}")
        ax.axvline(p5_v, color="crimson", linestyle=":", linewidth=1.5,
                   label=f"5th pct: {p5_v:.3f}")
        ax.set_xlabel("Terminal P&L", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.25)

    cost_label = f"{tc.cost_rate * 10_000:.0f} bps" if tc.cost_rate > 0 else "zero cost"
    display_name = _PAYOFF_METADATA.get(config.payoff.payoff_type, {}).get(
        "payoff_display_name", "GBM"
    )
    fig.suptitle(
        f"P&L Distribution — {display_name} ({cost_label})", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    plot_path = os.path.join(out_dir, "pnl_distribution.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


# ---------------------------------------------------------------------------
# Background run worker
# ---------------------------------------------------------------------------

def _run_worker(run_id: str, preset: dict[str, Any]) -> None:
    """Worker function executed in a background thread."""
    state = RUNS[run_id]
    out_dir = state.out_dir
    try:
        config: ExperimentConfig = dataclasses.replace(
            preset["config"], name=run_id
        )
        run_with_config(config, out_dir)
        _generate_pnl_plot(out_dir, config)

        with RUNS_LOCK:
            RUNS[run_id].status = "done"
            RUNS[run_id].completed_at = datetime.now(timezone.utc).isoformat()
    except Exception:
        err = traceback.format_exc()
        with RUNS_LOCK:
            RUNS[run_id].status = "error"
            RUNS[run_id].error = err
            RUNS[run_id].completed_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CreateRunRequest(BaseModel):
    preset_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/presets")
def list_presets() -> list[dict[str, Any]]:
    return _PRESET_LIST


@app.post("/api/runs", status_code=202)
def create_run(body: CreateRunRequest) -> dict[str, str]:
    preset_id = body.preset_id
    if preset_id not in PRESETS:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found")

    preset = PRESETS[preset_id]
    run_id = str(uuid4())
    out_dir = os.path.join("results", "runs", run_id)

    state = RunState(
        run_id=run_id,
        preset_id=preset_id,
        preset_name=preset["name"],
        status="running",
        created_at=datetime.now(timezone.utc).isoformat(),
        completed_at=None,
        error=None,
        out_dir=out_dir,
    )

    with RUNS_LOCK:
        RUNS[run_id] = state

    t = threading.Thread(target=_run_worker, args=(run_id, preset), daemon=True)
    t.start()

    return {"run_id": run_id, "status": "running"}


@app.get("/api/runs/{run_id}")
def get_run_status(run_id: str) -> dict[str, Any]:
    with RUNS_LOCK:
        state = RUNS.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return {
        "run_id": state.run_id,
        "preset_id": state.preset_id,
        "preset_name": state.preset_name,
        "status": state.status,
        "created_at": state.created_at,
        "completed_at": state.completed_at,
        "error": state.error,
    }


@app.get("/api/runs/{run_id}/results")
def get_run_results(run_id: str) -> dict[str, Any]:
    with RUNS_LOCK:
        state = RUNS.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    if state.status != "done":
        raise HTTPException(status_code=409, detail=f"Run status is '{state.status}', not 'done'")

    neural_path = os.path.join(state.out_dir, "neural_summary.json")
    classical_path = os.path.join(state.out_dir, "classical_summary.json")

    with open(neural_path) as f:
        neural = json.load(f)
    with open(classical_path) as f:
        classical = json.load(f)

    preset = PRESETS.get(state.preset_id, {})
    payoff_fields = {
        k: preset.get(k, "")
        for k in (
            "payoff_display_name", "payoff_formula", "payoff_profile",
            "delta_range", "benchmark_label", "classical_description",
        )
    }
    return {
        "run_id": run_id,
        "preset_id": state.preset_id,
        "preset_name": state.preset_name,
        **payoff_fields,
        "neural": neural,
        "classical": classical,
    }


@app.get("/api/runs/{run_id}/plot")
def get_run_plot(run_id: str) -> FileResponse:
    with RUNS_LOCK:
        state = RUNS.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    if state.status != "done":
        raise HTTPException(status_code=409, detail=f"Run status is '{state.status}', not 'done'")

    plot_path = os.path.join(state.out_dir, "pnl_distribution.png")
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="Plot not yet generated")

    return FileResponse(plot_path, media_type="image/png")


# ---------------------------------------------------------------------------
# Surface computation helper
# ---------------------------------------------------------------------------

# Grid constants (balance smoothness vs. response size)
_N_S = 40    # stock-price axis points
_N_TAU = 30  # time-to-maturity axis points


def _compute_surface_data(out_dir: str) -> dict[str, Any]:
    """Compute classical, neural, and difference hedge surfaces over a (S, τ) grid.

    Design notes
    ------------
    - Previous-position (prev_delta) is fixed at 0 for all grid points.
      This gives a clean "what would the network do starting from flat?"
      slice through the three-dimensional state space.  The assumption is
      clearly communicated in the returned metadata.
    - Stock axis spans [0.6·S₀, 1.6·S₀] — deep ITM to deep OTM.
    - τ axis spans [T/n_steps, T] — one step before expiry to full maturity.
    - Grid size: _N_S × _N_TAU = 1200 points; ~0.1 s for 32-dim HedgeNet.
    - The classical benchmark delta is payoff-aware: call → N(d₁),
      put → N(d₁)−1, spread → net delta, straddle → 2N(d₁)−1.
    """
    config_path = os.path.join(out_dir, "config.json")
    checkpoint_path = os.path.join(out_dir, "model_checkpoint.pt")

    with open(config_path) as f:
        cfg = json.load(f)

    mc = MarketConfig(**cfg["market"])
    tc = TrainingConfig(**cfg["training"])
    pc = PayoffConfig(**cfg["payoff"])

    spec = get_payoff_spec(pc, mc)

    # Build grid
    s_vals = np.linspace(0.6 * mc.s0, 1.6 * mc.s0, _N_S)
    tau_vals = np.linspace(mc.T / mc.n_steps, mc.T, _N_TAU)
    S, TAU = np.meshgrid(s_vals, tau_vals)  # both shape (_N_TAU, _N_S)

    # Classical delta surface (payoff-aware, vectorised)
    Z_bs = np.asarray(spec.classical_delta_fn(S, TAU), dtype=float)

    # Neural hedge surface — prev_delta fixed at 0, delta transform applied
    model = HedgeNet(n_layers=tc.n_layers, hidden_dim=tc.hidden_dim)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    with torch.no_grad():
        s_norm = torch.tensor((S / mc.s0).flatten(), dtype=torch.float32)
        tau_flat = torch.tensor(TAU.flatten(), dtype=torch.float32)
        prev_delta = torch.zeros(_N_S * _N_TAU, dtype=torch.float32)
        feats = torch.stack([s_norm, tau_flat, prev_delta], dim=1)
        raw = model(feats)
        Z_neural = spec.delta_transform(raw).numpy().reshape(_N_TAU, _N_S)

    Z_diff = Z_neural - Z_bs

    return {
        "x": s_vals.tolist(),
        "y": tau_vals.tolist(),
        "z_bs": Z_bs.tolist(),
        "z_neural": Z_neural.tolist(),
        "z_diff": Z_diff.tolist(),
        "x_label": f"Stock Price  (K = {mc.k:.0f})",
        "y_label": "Time to Maturity τ (years)",
        "z_label": "Hedge Ratio Δ",
        "metadata": {
            "s0": mc.s0,
            "k": mc.k,
            "r": mc.r,
            "sigma": mc.sigma,
            "T": mc.T,
            "cost_rate": tc.cost_rate,
            "payoff_type": pc.payoff_type,
            "classical_description": spec.classical_description,
            "n_s": _N_S,
            "n_tau": _N_TAU,
            "z_min_bs": float(Z_bs.min()),
            "z_max_bs": float(Z_bs.max()),
            "prev_delta_assumption": (
                "fixed at 0 — static slice: what would the neural hedger do "
                "starting from a flat (unhedged) position?"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Surface endpoint
# ---------------------------------------------------------------------------

@app.get("/api/runs/{run_id}/surface")
def get_run_surface(run_id: str) -> dict[str, Any]:
    """Return BS, neural, and difference hedge surfaces for a completed run.

    Response shape
    --------------
    x         : list[float]         stock prices, length n_s
    y         : list[float]         times to maturity, length n_tau
    z_bs      : list[list[float]]   shape (n_tau, n_s), Black-Scholes delta
    z_neural  : list[list[float]]   shape (n_tau, n_s), neural hedge ratio
    z_diff    : list[list[float]]   shape (n_tau, n_s), neural − BS
    x_label   : str
    y_label   : str
    z_label   : str
    metadata  : dict                grid bounds and fixed-slice assumption
    """
    with RUNS_LOCK:
        state = RUNS.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    if state.status != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Run status is '{state.status}'; surface available only after 'done'",
        )

    return _compute_surface_data(state.out_dir)
