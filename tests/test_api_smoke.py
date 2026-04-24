"""Smoke tests for the FastAPI backend (no live server required).

Uses FastAPI's TestClient to exercise routes without starting uvicorn.
These tests are fast (no experiment runs triggered).
"""

from __future__ import annotations

import sys
import os

import pytest

# Ensure repo root is on path (may differ by test runner cwd)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from services.sim.main import app

client = TestClient(app)


def test_health_ok() -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_presets_returns_list() -> None:
    res = client.get("/api/presets")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 2


def test_presets_have_required_fields() -> None:
    res = client.get("/api/presets")
    for preset in res.json():
        for field in ["id", "name", "description", "cost_rate", "est_seconds"]:
            assert field in preset, f"Missing field '{field}' in preset {preset.get('id')}"


def test_presets_no_config_field_exposed() -> None:
    """The internal ExperimentConfig must not be serialised and returned to clients."""
    res = client.get("/api/presets")
    for preset in res.json():
        assert "config" not in preset


def test_create_run_unknown_preset_returns_404() -> None:
    res = client.post("/api/runs", json={"preset_id": "does_not_exist_xyz"})
    assert res.status_code == 404


def test_get_run_unknown_id_returns_404() -> None:
    res = client.get("/api/runs/00000000-0000-0000-0000-000000000000")
    assert res.status_code == 404


def test_get_run_results_unknown_id_returns_404() -> None:
    res = client.get("/api/runs/00000000-0000-0000-0000-000000000000/results")
    assert res.status_code == 404


def test_get_run_plot_unknown_id_returns_404() -> None:
    res = client.get("/api/runs/00000000-0000-0000-0000-000000000000/plot")
    assert res.status_code == 404


def test_create_run_returns_run_id_and_status() -> None:
    """POST /api/runs creates a run and returns a run_id without blocking."""
    import threading
    # We don't want the background thread to actually run a full experiment in tests.
    # Patch the worker to do nothing.
    from services.sim import main as sim_main

    original_worker = sim_main._run_worker

    def noop_worker(run_id: str, preset: dict) -> None:  # type: ignore[type-arg]
        pass

    sim_main._run_worker = noop_worker  # type: ignore[assignment]
    try:
        res = client.post("/api/runs", json={"preset_id": "gbm_call_zero_cost"})
        assert res.status_code == 202
        body = res.json()
        assert "run_id" in body
        assert body["status"] == "running"
    finally:
        sim_main._run_worker = original_worker  # type: ignore[assignment]


def test_get_run_status_after_create() -> None:
    """After creating a run, GET /api/runs/{id} should return the run state."""
    from services.sim import main as sim_main

    original_worker = sim_main._run_worker

    def noop_worker(run_id: str, preset: dict) -> None:  # type: ignore[type-arg]
        pass

    sim_main._run_worker = noop_worker  # type: ignore[assignment]
    try:
        create_res = client.post("/api/runs", json={"preset_id": "gbm_call_5bps"})
        run_id = create_res.json()["run_id"]

        status_res = client.get(f"/api/runs/{run_id}")
        assert status_res.status_code == 200
        body = status_res.json()
        assert body["run_id"] == run_id
        assert body["preset_id"] == "gbm_call_5bps"
        assert body["status"] in ("running", "done", "error")
    finally:
        sim_main._run_worker = original_worker  # type: ignore[assignment]


def test_get_run_results_before_done_returns_409() -> None:
    """Results endpoint must refuse when status != done."""
    from services.sim import main as sim_main

    original_worker = sim_main._run_worker

    def noop_worker(run_id: str, preset: dict) -> None:  # type: ignore[type-arg]
        pass

    sim_main._run_worker = noop_worker  # type: ignore[assignment]
    try:
        create_res = client.post("/api/runs", json={"preset_id": "gbm_call_zero_cost"})
        run_id = create_res.json()["run_id"]

        results_res = client.get(f"/api/runs/{run_id}/results")
        # Status is still "running" (noop worker never sets it to done)
        assert results_res.status_code == 409
    finally:
        sim_main._run_worker = original_worker  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Surface endpoint tests
# ---------------------------------------------------------------------------

def test_surface_unknown_run_returns_404() -> None:
    res = client.get("/api/runs/00000000-0000-0000-0000-000000000000/surface")
    assert res.status_code == 404


def test_surface_not_done_returns_409() -> None:
    from services.sim import main as sim_main

    original_worker = sim_main._run_worker

    def noop_worker(run_id: str, preset: dict) -> None:  # type: ignore[type-arg]
        pass

    sim_main._run_worker = noop_worker  # type: ignore[assignment]
    try:
        create_res = client.post("/api/runs", json={"preset_id": "gbm_call_zero_cost"})
        run_id = create_res.json()["run_id"]
        surface_res = client.get(f"/api/runs/{run_id}/surface")
        assert surface_res.status_code == 409
    finally:
        sim_main._run_worker = original_worker  # type: ignore[assignment]


def test_surface_endpoint_returns_correct_shape() -> None:
    """Full round-trip: tiny config → surface endpoint → validate grid shape."""
    import dataclasses, shutil, tempfile, json, torch
    from services.sim import main as sim_main
    from src.experiments.configs import (
        ExperimentConfig, MarketConfig, PayoffConfig,
        HedgeUniverseConfig, TrainingConfig, EvaluationConfig,
    )
    from src.experiments.runner import run_with_config
    from src.experiments.summaries import SummaryRow

    tiny_cfg = ExperimentConfig(
        name="_surface_test",
        market=MarketConfig(n_steps=5),
        payoff=PayoffConfig(payoff_type="call"),
        hedge_universe=HedgeUniverseConfig(universe_type="stock_only"),
        training=TrainingConfig(n_paths=256, n_epochs=2, hidden_dim=8, n_layers=2, seed=7, cost_rate=0.0),
        evaluation=EvaluationConfig(n_paths=200, seed_offset=1000),
    )

    with tempfile.TemporaryDirectory() as out_dir:
        run_with_config(tiny_cfg, out_dir)

        surface = sim_main._compute_surface_data(out_dir)

    n_s = sim_main._N_S
    n_tau = sim_main._N_TAU

    assert len(surface["x"]) == n_s
    assert len(surface["y"]) == n_tau
    for key in ("z_bs", "z_neural", "z_diff"):
        z = surface[key]
        assert len(z) == n_tau, f"{key}: expected {n_tau} rows, got {len(z)}"
        assert len(z[0]) == n_s, f"{key}: expected {n_s} cols, got {len(z[0])}"


def test_surface_bs_values_in_unit_interval() -> None:
    """Black-Scholes delta surface values must lie in [0, 1] for a European call."""
    import dataclasses, tempfile
    from services.sim import main as sim_main
    from src.experiments.configs import (
        ExperimentConfig, MarketConfig, PayoffConfig,
        HedgeUniverseConfig, TrainingConfig, EvaluationConfig,
    )
    from src.experiments.runner import run_with_config

    tiny_cfg = ExperimentConfig(
        name="_surface_bs_test",
        market=MarketConfig(n_steps=5),
        payoff=PayoffConfig(payoff_type="call"),
        hedge_universe=HedgeUniverseConfig(universe_type="stock_only"),
        training=TrainingConfig(n_paths=256, n_epochs=2, hidden_dim=8, n_layers=2, seed=8, cost_rate=0.0),
        evaluation=EvaluationConfig(n_paths=200, seed_offset=1000),
    )

    with tempfile.TemporaryDirectory() as out_dir:
        run_with_config(tiny_cfg, out_dir)
        surface = sim_main._compute_surface_data(out_dir)

    import numpy as np
    z_bs = np.array(surface["z_bs"])
    assert float(z_bs.min()) >= -1e-6, f"BS delta below 0: {z_bs.min()}"
    assert float(z_bs.max()) <= 1 + 1e-6, f"BS delta above 1: {z_bs.max()}"


def test_surface_neural_values_in_unit_interval() -> None:
    """Neural hedge ratio surface values must lie in (0, 1) — HedgeNet has sigmoid output."""
    import tempfile
    from services.sim import main as sim_main
    from src.experiments.configs import (
        ExperimentConfig, MarketConfig, PayoffConfig,
        HedgeUniverseConfig, TrainingConfig, EvaluationConfig,
    )
    from src.experiments.runner import run_with_config

    tiny_cfg = ExperimentConfig(
        name="_surface_neural_test",
        market=MarketConfig(n_steps=5),
        payoff=PayoffConfig(payoff_type="call"),
        hedge_universe=HedgeUniverseConfig(universe_type="stock_only"),
        training=TrainingConfig(n_paths=256, n_epochs=2, hidden_dim=8, n_layers=2, seed=9, cost_rate=0.0005),
        evaluation=EvaluationConfig(n_paths=200, seed_offset=1000),
    )

    with tempfile.TemporaryDirectory() as out_dir:
        run_with_config(tiny_cfg, out_dir)
        surface = sim_main._compute_surface_data(out_dir)

    import numpy as np
    z_neural = np.array(surface["z_neural"])
    assert float(z_neural.min()) >= 0.0, f"Neural delta below 0: {z_neural.min()}"
    assert float(z_neural.max()) <= 1.0, f"Neural delta above 1: {z_neural.max()}"


def test_surface_diff_equals_neural_minus_bs() -> None:
    """z_diff must equal z_neural - z_bs element-wise."""
    import tempfile
    from services.sim import main as sim_main
    from src.experiments.configs import (
        ExperimentConfig, MarketConfig, PayoffConfig,
        HedgeUniverseConfig, TrainingConfig, EvaluationConfig,
    )
    from src.experiments.runner import run_with_config
    import numpy as np

    tiny_cfg = ExperimentConfig(
        name="_surface_diff_test",
        market=MarketConfig(n_steps=5),
        payoff=PayoffConfig(payoff_type="call"),
        hedge_universe=HedgeUniverseConfig(universe_type="stock_only"),
        training=TrainingConfig(n_paths=256, n_epochs=2, hidden_dim=8, n_layers=2, seed=11, cost_rate=0.0),
        evaluation=EvaluationConfig(n_paths=200, seed_offset=1000),
    )

    with tempfile.TemporaryDirectory() as out_dir:
        run_with_config(tiny_cfg, out_dir)
        surface = sim_main._compute_surface_data(out_dir)

    z_bs = np.array(surface["z_bs"])
    z_neural = np.array(surface["z_neural"])
    z_diff = np.array(surface["z_diff"])
    np.testing.assert_allclose(z_diff, z_neural - z_bs, atol=1e-6)


# ---------------------------------------------------------------------------
# Phase 3 preset tests
# ---------------------------------------------------------------------------

def test_presets_include_new_payoff_types() -> None:
    """Phase 3 presets for put, spread, and straddle must appear."""
    res = client.get("/api/presets")
    ids = {p["id"] for p in res.json()}
    assert "gbm_put_5bps" in ids, f"gbm_put_5bps missing from {ids}"
    assert "gbm_call_spread_5bps" in ids, f"gbm_call_spread_5bps missing from {ids}"
    assert "gbm_straddle_5bps" in ids, f"gbm_straddle_5bps missing from {ids}"


def test_presets_total_count_at_least_five() -> None:
    res = client.get("/api/presets")
    assert len(res.json()) >= 5


def test_put_preset_fields() -> None:
    res = client.get("/api/presets")
    put_preset = next((p for p in res.json() if p["id"] == "gbm_put_5bps"), None)
    assert put_preset is not None
    assert put_preset["cost_rate"] == pytest.approx(0.0005)
    assert "Put" in put_preset["payoff_type"] or "put" in put_preset["payoff_type"].lower()


def test_spread_preset_fields() -> None:
    res = client.get("/api/presets")
    spread_preset = next((p for p in res.json() if p["id"] == "gbm_call_spread_5bps"), None)
    assert spread_preset is not None
    assert spread_preset["cost_rate"] == pytest.approx(0.0005)


def test_straddle_preset_fields() -> None:
    res = client.get("/api/presets")
    straddle_preset = next((p for p in res.json() if p["id"] == "gbm_straddle_5bps"), None)
    assert straddle_preset is not None
    assert straddle_preset["cost_rate"] == pytest.approx(0.0005)


def test_create_put_run_returns_202() -> None:
    from services.sim import main as sim_main
    original = sim_main._run_worker
    sim_main._run_worker = lambda run_id, preset: None  # type: ignore[assignment]
    try:
        res = client.post("/api/runs", json={"preset_id": "gbm_put_5bps"})
        assert res.status_code == 202
        assert "run_id" in res.json()
    finally:
        sim_main._run_worker = original  # type: ignore[assignment]


def test_create_spread_run_returns_202() -> None:
    from services.sim import main as sim_main
    original = sim_main._run_worker
    sim_main._run_worker = lambda run_id, preset: None  # type: ignore[assignment]
    try:
        res = client.post("/api/runs", json={"preset_id": "gbm_call_spread_5bps"})
        assert res.status_code == 202
    finally:
        sim_main._run_worker = original  # type: ignore[assignment]


def test_create_straddle_run_returns_202() -> None:
    from services.sim import main as sim_main
    original = sim_main._run_worker
    sim_main._run_worker = lambda run_id, preset: None  # type: ignore[assignment]
    try:
        res = client.post("/api/runs", json={"preset_id": "gbm_straddle_5bps"})
        assert res.status_code == 202
    finally:
        sim_main._run_worker = original  # type: ignore[assignment]
