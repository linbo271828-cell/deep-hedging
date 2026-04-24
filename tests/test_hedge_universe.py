"""Smoke tests for stock + option hedge universe."""
import math

import numpy as np
import pytest
import torch

from src.hedging.hedge_universe import StockOnlyUniverse, StockPlusOptionUniverse, make_universe
from src.neural.architectures import HedgeNetMulti
from src.experiments.configs import MarketConfig, HedgeUniverseConfig

MC = MarketConfig()
UC = HedgeUniverseConfig(universe_type="stock_plus_option", hedge_option_strike=100.0, hedge_option_maturity=0.5)

# ── StockPlusOptionUniverse ──

def test_stock_only_universe_shape():
    u = StockOnlyUniverse()
    paths = np.ones((50, 51))
    tg = np.linspace(0, 0.5, 51)
    prices = u.instrument_prices(paths, tg)
    assert prices.shape == (50, 51, 1)

def test_spo_universe_hedge_price_positive():
    u = StockPlusOptionUniverse(MC, UC)
    price = u.hedge_option_price(np.array([100.0]), 0.25)
    assert float(price[0]) > 0

def test_spo_universe_hedge_price_at_expiry():
    u = StockPlusOptionUniverse(MC, UC)
    # ITM at tau=0: value = intrinsic
    price_itm = u.hedge_option_price(np.array([120.0]), 0.0)
    assert abs(float(price_itm[0]) - 20.0) < 1e-6
    # OTM at tau=0: value = 0
    price_otm = u.hedge_option_price(np.array([80.0]), 0.0)
    assert float(price_otm[0]) == pytest.approx(0.0)

def test_spo_prices_along_paths_shape():
    u = StockPlusOptionUniverse(MC, UC)
    n_paths, n_steps = 20, 10
    paths = np.random.default_rng(1).lognormal(size=(n_paths, n_steps+1)) * 100
    tg = np.linspace(0, 0.5, n_steps+1)
    hp = u.hedge_option_prices_along_paths(paths, tg)
    assert hp.shape == (n_paths, n_steps+1)
    assert np.all(hp >= 0)

def test_spo_instrument_prices_shape():
    u = StockPlusOptionUniverse(MC, UC)
    paths = np.ones((10, 11)) * 100
    tg = np.linspace(0, 0.5, 11)
    ip = u.instrument_prices(paths, tg)
    assert ip.shape == (10, 11, 2)
    assert np.allclose(ip[:, :, 0], paths)

def test_make_universe_stock_only():
    u = make_universe(MC, HedgeUniverseConfig(universe_type="stock_only"))
    assert u.n_instruments == 1

def test_make_universe_stock_plus_option():
    u = make_universe(MC, UC)
    assert u.n_instruments == 2

def test_make_universe_invalid():
    import dataclasses
    bad = dataclasses.replace(UC, universe_type="invalid")  # type: ignore
    with pytest.raises(ValueError):
        make_universe(MC, bad)  # type: ignore

# ── HedgeNetMulti ──

def test_hedge_net_multi_output_shape():
    model = HedgeNetMulti(n_layers=2, hidden_dim=8)
    x = torch.randn(32, 5)
    out = model(x)
    assert out.shape == (32, 2)

def test_hedge_net_multi_output_range():
    model = HedgeNetMulti(n_layers=2, hidden_dim=8)
    x = torch.randn(100, 5)
    out = model(x)
    assert bool((out >= 0).all() and (out <= 1).all())

# ── hedge_pnl_multi ──

def test_hedge_pnl_multi_shape():
    from src.hedging.pnl import hedge_pnl_multi
    from src.models.gbm import GBM, time_grid as make_tg
    n_paths, n_steps = 50, 10
    gbm = GBM(s0=100, mu=0.02, sigma=0.2)
    paths = gbm.sample_paths(n_paths, n_steps, 0.5, seed=1)
    tg = make_tg(n_steps, 0.5)
    u = StockPlusOptionUniverse(MC, UC)
    hp = u.hedge_option_prices_along_paths(paths, tg)
    pnl = hedge_pnl_multi(
        paths=paths,
        hedge_option_prices=hp,
        stock_delta_fn=lambda s, tau: np.zeros(len(s)),
        hedge_delta_fn=lambda s, tau: np.zeros(len(s)),
        time_grid=tg,
        payoff_fn=lambda s: np.maximum(s - 100.0, 0.0),
        initial_option_price=5.0,
        cost_rate=0.0,
        r=0.0,
        k_h=100.0,
    )
    assert pnl.shape == (n_paths,)

def test_hedge_pnl_multi_zero_hedge_matches_pnl():
    """With zero hedge positions, hedge_pnl_multi should give same result as holding no hedges."""
    from src.hedging.pnl import hedge_pnl_multi
    from src.models.gbm import GBM, time_grid as make_tg
    n_paths, n_steps = 100, 10
    gbm = GBM(s0=100, mu=0.02, sigma=0.2)
    paths = gbm.sample_paths(n_paths, n_steps, 0.5, seed=42)
    tg = make_tg(n_steps, 0.5)
    u = StockPlusOptionUniverse(MC, UC)
    hp = u.hedge_option_prices_along_paths(paths, tg)
    pnl = hedge_pnl_multi(
        paths=paths,
        hedge_option_prices=hp,
        stock_delta_fn=lambda s, tau: np.zeros(len(s)),
        hedge_delta_fn=lambda s, tau: np.zeros(len(s)),
        time_grid=tg,
        payoff_fn=lambda s: np.maximum(s - 100.0, 0.0),
        initial_option_price=5.0,
        cost_rate=0.0,
        r=0.0,
        k_h=100.0,
    )
    # With zero hedging and zero cost, P&L = premium - payoff
    s_T = paths[:, n_steps]
    expected = 5.0 - np.maximum(s_T - 100.0, 0.0)
    np.testing.assert_allclose(pnl, expected, atol=1e-9)

# ── Runner smoke (stock+option) ──

def test_runner_stock_option_call():
    """smoke: stock+option runner produces two SummaryRow entries without crashing."""
    import tempfile
    from src.experiments.configs import (
        ExperimentConfig, PayoffConfig, TrainingConfig, EvaluationConfig
    )
    from src.experiments.runner import run_with_config
    cfg = ExperimentConfig(
        name="_test_spo",
        market=MarketConfig(n_steps=5),
        payoff=PayoffConfig(payoff_type="call"),
        hedge_universe=HedgeUniverseConfig(
            universe_type="stock_plus_option",
            hedge_option_strike=100.0,
            hedge_option_maturity=0.5,
        ),
        training=TrainingConfig(n_paths=128, n_epochs=2, hidden_dim=8, n_layers=2, seed=99, cost_rate=0.0005),
        evaluation=EvaluationConfig(n_paths=100, seed_offset=1000),
    )
    with tempfile.TemporaryDirectory() as d:
        rows = run_with_config(cfg, d)
    assert len(rows) == 2
    assert rows[0].strategy == "neural"
    assert rows[1].strategy == "classical"
    assert rows[0].hedge_universe == "stock_plus_option"
    assert math.isfinite(rows[0].cvar_95)
    assert math.isfinite(rows[1].cvar_95)

# ── API presets ──

def test_api_presets_include_stock_option():
    from fastapi.testclient import TestClient
    from services.sim.main import app
    client = TestClient(app)
    res = client.get("/api/presets")
    ids = {p["id"] for p in res.json()}
    assert "gbm_call_stock_option_5bps" in ids
    assert "gbm_straddle_stock_option_5bps" in ids

def test_api_presets_total_count():
    from fastapi.testclient import TestClient
    from services.sim.main import app
    client = TestClient(app)
    res = client.get("/api/presets")
    assert len(res.json()) >= 7

def test_api_stock_option_preset_fields():
    from fastapi.testclient import TestClient
    from services.sim.main import app
    client = TestClient(app)
    res = client.get("/api/presets")
    spo = next((p for p in res.json() if p["id"] == "gbm_call_stock_option_5bps"), None)
    assert spo is not None
    assert spo["hedge_universe_display"] == "Stock + ATM Call Hedge"
    assert spo["cost_rate"] == pytest.approx(0.0005)

def test_api_create_stock_option_run():
    from fastapi.testclient import TestClient
    from services.sim import main as sim_main
    from services.sim.main import app
    client = TestClient(app)
    orig = sim_main._run_worker
    sim_main._run_worker = lambda run_id, preset: None  # type: ignore
    try:
        res = client.post("/api/runs", json={"preset_id": "gbm_call_stock_option_5bps"})
        assert res.status_code == 202
        assert "run_id" in res.json()
    finally:
        sim_main._run_worker = orig  # type: ignore

if __name__ == "__main__":
    import subprocess, sys
    sys.exit(subprocess.call(["python", "-m", "pytest", __file__, "-v"]))
