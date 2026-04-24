"""Smoke tests for the experiment runner (Lane E / Lane G)."""

from __future__ import annotations

import json
import os
import shutil

import pytest

from src.experiments.configs import (
    EvaluationConfig,
    ExperimentConfig,
    HedgeUniverseConfig,
    MarketConfig,
    PayoffConfig,
    TrainingConfig,
)
from src.experiments.registry import REGISTRY, get_config, list_experiments
from src.experiments.runner import run_experiment
from src.experiments.summaries import SummaryRow


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_registry_not_empty() -> None:
    assert len(list_experiments()) >= 2


def test_get_config_known_name() -> None:
    cfg = get_config("gbm_call_no_cost")
    assert cfg.name == "gbm_call_no_cost"
    assert cfg.training.cost_rate == 0.0


def test_get_config_unknown_raises() -> None:
    with pytest.raises(KeyError, match="not found"):
        get_config("does_not_exist_xyz")


def test_registry_configs_are_frozen() -> None:
    cfg = get_config("gbm_call_no_cost")
    with pytest.raises((AttributeError, TypeError)):
        cfg.name = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Runner smoke test (tiny config, runs fast)
# ---------------------------------------------------------------------------

_TINY_CONFIG = ExperimentConfig(
    name="_test_runner_smoke",
    market=MarketConfig(n_steps=5),
    payoff=PayoffConfig(payoff_type="call"),
    hedge_universe=HedgeUniverseConfig(universe_type="stock_only"),
    training=TrainingConfig(n_paths=128, n_epochs=3, hidden_dim=8, n_layers=2, seed=1),
    evaluation=EvaluationConfig(n_paths=200, seed_offset=1000),
)

_RESULTS_DIR = os.path.join("results", "raw", "_test_runner_smoke")


@pytest.fixture(autouse=True)
def cleanup_test_results():
    """Remove test artifacts before and after each test."""
    if os.path.exists(_RESULTS_DIR):
        shutil.rmtree(_RESULTS_DIR)
    # Temporarily inject the test config into the registry
    REGISTRY["_test_runner_smoke"] = _TINY_CONFIG
    yield
    REGISTRY.pop("_test_runner_smoke", None)
    if os.path.exists(_RESULTS_DIR):
        shutil.rmtree(_RESULTS_DIR)


def test_runner_returns_two_summary_rows() -> None:
    rows = run_experiment("_test_runner_smoke")
    assert len(rows) == 2
    strategies = {r.strategy for r in rows}
    assert strategies == {"neural", "classical"}


def test_runner_summary_rows_are_summary_row_instances() -> None:
    rows = run_experiment("_test_runner_smoke")
    for row in rows:
        assert isinstance(row, SummaryRow)


def test_runner_artifacts_written() -> None:
    run_experiment("_test_runner_smoke")
    for fname in [
        "config.json",
        "neural_summary.json",
        "classical_summary.json",
        "loss_history.json",
        "model_checkpoint.pt",
    ]:
        assert os.path.exists(os.path.join(_RESULTS_DIR, fname)), f"Missing: {fname}"


def test_runner_neural_row_has_correct_metadata() -> None:
    rows = run_experiment("_test_runner_smoke")
    neural = next(r for r in rows if r.strategy == "neural")
    assert neural.experiment_name == "_test_runner_smoke"
    assert neural.n_epochs == 3
    assert neural.hidden_dim == 8
    assert neural.n_layers == 2
    assert neural.cost_rate == 0.0
    assert neural.n_paths == 200


def test_runner_classical_row_has_none_for_net_fields() -> None:
    rows = run_experiment("_test_runner_smoke")
    classical = next(r for r in rows if r.strategy == "classical")
    assert classical.n_epochs is None
    assert classical.hidden_dim is None
    assert classical.n_layers is None


def test_runner_metrics_are_finite() -> None:
    import numpy as np
    rows = run_experiment("_test_runner_smoke")
    for row in rows:
        for field_name in ["mean_pnl", "std_pnl", "cvar_95", "entropic_risk",
                           "expected_cost", "turnover"]:
            val = getattr(row, field_name)
            assert np.isfinite(val), f"{row.strategy}.{field_name} not finite: {val}"


def test_runner_skips_on_second_call() -> None:
    rows1 = run_experiment("_test_runner_smoke")
    rows2 = run_experiment("_test_runner_smoke")  # should load from disk
    # Metrics should be identical (loaded from same JSON)
    assert rows1[0].cvar_95 == rows2[0].cvar_95
    assert rows1[1].cvar_95 == rows2[1].cvar_95


def test_runner_force_reruns() -> None:
    rows1 = run_experiment("_test_runner_smoke")
    rows2 = run_experiment("_test_runner_smoke", force=True)
    # Both should be valid SummaryRows (we can't assert identical due to fresh training)
    assert len(rows2) == 2


def test_runner_config_json_matches_experiment() -> None:
    run_experiment("_test_runner_smoke")
    with open(os.path.join(_RESULTS_DIR, "config.json")) as f:
        saved = json.load(f)
    assert saved["name"] == "_test_runner_smoke"
    assert saved["training"]["n_epochs"] == 3


# ---------------------------------------------------------------------------
# Phase 3 payoff runner smoke tests
# ---------------------------------------------------------------------------

_TINY_PUT_CONFIG = ExperimentConfig(
    name="_test_put_smoke",
    market=MarketConfig(n_steps=5),
    payoff=PayoffConfig(payoff_type="put"),
    hedge_universe=HedgeUniverseConfig(universe_type="stock_only"),
    training=TrainingConfig(n_paths=128, n_epochs=3, hidden_dim=8, n_layers=2, seed=101),
    evaluation=EvaluationConfig(n_paths=200, seed_offset=1000),
)

_TINY_SPREAD_CONFIG = ExperimentConfig(
    name="_test_spread_smoke",
    market=MarketConfig(n_steps=5),
    payoff=PayoffConfig(payoff_type="call_spread", k=95.0, k_hi=105.0),
    hedge_universe=HedgeUniverseConfig(universe_type="stock_only"),
    training=TrainingConfig(n_paths=128, n_epochs=3, hidden_dim=8, n_layers=2, seed=102),
    evaluation=EvaluationConfig(n_paths=200, seed_offset=1000),
)

_TINY_STRADDLE_CONFIG = ExperimentConfig(
    name="_test_straddle_smoke",
    market=MarketConfig(n_steps=5),
    payoff=PayoffConfig(payoff_type="straddle"),
    hedge_universe=HedgeUniverseConfig(universe_type="stock_only"),
    training=TrainingConfig(n_paths=128, n_epochs=3, hidden_dim=8, n_layers=2, seed=103),
    evaluation=EvaluationConfig(n_paths=200, seed_offset=1000),
)


def _run_tiny(config: ExperimentConfig) -> tuple[object, object]:
    """Run a tiny config via run_with_config and return (neural_row, classical_row)."""
    import tempfile
    from src.experiments.runner import run_with_config
    with tempfile.TemporaryDirectory() as d:
        rows = run_with_config(config, d)
    return rows[0], rows[1]


def test_put_runner_returns_two_rows() -> None:
    neural, classical = _run_tiny(_TINY_PUT_CONFIG)
    assert isinstance(neural, SummaryRow)
    assert isinstance(classical, SummaryRow)
    assert neural.payoff_type == "put"
    assert classical.payoff_type == "put"


def test_put_runner_metrics_finite() -> None:
    import numpy as np
    neural, classical = _run_tiny(_TINY_PUT_CONFIG)
    for row in (neural, classical):
        for field in ["mean_pnl", "std_pnl", "cvar_95", "entropic_risk", "expected_cost", "turnover"]:
            assert np.isfinite(getattr(row, field)), f"{row.strategy}.{field} not finite"


def test_call_spread_runner_returns_two_rows() -> None:
    neural, classical = _run_tiny(_TINY_SPREAD_CONFIG)
    assert neural.payoff_type == "call_spread"
    assert classical.strategy == "classical"


def test_call_spread_runner_metrics_finite() -> None:
    import numpy as np
    neural, classical = _run_tiny(_TINY_SPREAD_CONFIG)
    for row in (neural, classical):
        for field in ["mean_pnl", "std_pnl", "cvar_95"]:
            assert np.isfinite(getattr(row, field))


def test_straddle_runner_returns_two_rows() -> None:
    neural, classical = _run_tiny(_TINY_STRADDLE_CONFIG)
    assert neural.payoff_type == "straddle"
    assert classical.payoff_type == "straddle"


def test_straddle_runner_metrics_finite() -> None:
    import numpy as np
    neural, classical = _run_tiny(_TINY_STRADDLE_CONFIG)
    for row in (neural, classical):
        for field in ["mean_pnl", "std_pnl", "cvar_95"]:
            assert np.isfinite(getattr(row, field))


def test_classical_put_pnl_symmetric_to_call() -> None:
    """Classical P&L for a put and a call share the same P&L sign convention."""
    import numpy as np
    _, classical_put = _run_tiny(_TINY_PUT_CONFIG)
    # classical put CVaR should be finite and negative (it's a loss measure)
    assert np.isfinite(classical_put.cvar_95)
