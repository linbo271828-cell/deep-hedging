"""Smoke tests for neural hedger training and evaluation (Lane D / Lane G)."""

from __future__ import annotations

import numpy as np
import torch

from src.experiments.configs import EvaluationConfig, MarketConfig, TrainingConfig
from src.neural.architectures import HedgeNet
from src.neural.evaluate import EvaluationResult, evaluate
from src.neural.train import train


# Small config for fast smoke testing (< 5 s total)
_MARKET = MarketConfig(n_steps=10)
_TRAINING = TrainingConfig(n_paths=512, n_epochs=5, hidden_dim=16, n_layers=2, seed=99)
_EVAL = EvaluationConfig(n_paths=1_000, seed_offset=1000)


def _trained_model() -> HedgeNet:
    model = HedgeNet(n_layers=_TRAINING.n_layers, hidden_dim=_TRAINING.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=_TRAINING.lr)
    model, _ = train(_TRAINING, _MARKET, model, optimizer)
    return model


def test_hedgenet_forward_shape() -> None:
    model = HedgeNet(n_layers=2, hidden_dim=16)
    x = torch.randn(32, 3)
    out = model(x)
    assert out.shape == (32,)
    assert bool((out >= 0).all() and (out <= 1).all())


def test_train_returns_model_and_history() -> None:
    model = HedgeNet(n_layers=_TRAINING.n_layers, hidden_dim=_TRAINING.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=_TRAINING.lr)
    trained, history = train(_TRAINING, _MARKET, model, optimizer)
    assert isinstance(trained, HedgeNet)
    assert len(history) == _TRAINING.n_epochs
    assert all(isinstance(v, float) for v in history)


def test_train_loss_decreases() -> None:
    """CVaR loss should decrease over training (on average)."""
    model = HedgeNet(n_layers=2, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tc = TrainingConfig(n_paths=2_048, n_epochs=30, hidden_dim=32, n_layers=2, seed=7)
    _, history = train(tc, _MARKET, model, optimizer)
    # First-half mean loss should exceed second-half mean loss
    mid = len(history) // 2
    assert np.mean(history[:mid]) > np.mean(history[mid:]), (
        "Training loss did not decrease: first_half={:.4f}, second_half={:.4f}".format(
            float(np.mean(history[:mid])), float(np.mean(history[mid:]))
        )
    )


def test_evaluate_returns_evaluation_result() -> None:
    model = _trained_model()
    result = evaluate(model, _MARKET, _TRAINING, _EVAL)
    assert isinstance(result, EvaluationResult)
    assert result.n_paths == _EVAL.n_paths


def test_evaluate_metrics_are_finite() -> None:
    model = _trained_model()
    result = evaluate(model, _MARKET, _TRAINING, _EVAL)
    for field_name, val in result.__dict__.items():
        assert np.isfinite(val), f"metric '{field_name}' is not finite: {val}"


def test_evaluate_expected_cost_nonnegative() -> None:
    """Transaction costs are always non-negative."""
    model = _trained_model()
    result = evaluate(model, _MARKET, _TRAINING, _EVAL)
    assert result.expected_cost >= 0.0


def test_evaluate_zero_cost_rate_has_zero_expected_cost() -> None:
    tc_no_cost = TrainingConfig(
        n_paths=512, n_epochs=5, hidden_dim=16, n_layers=2, seed=99, cost_rate=0.0
    )
    model = HedgeNet(n_layers=2, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model, _ = train(tc_no_cost, _MARKET, model, optimizer)
    result = evaluate(model, _MARKET, tc_no_cost, _EVAL)
    assert result.expected_cost == 0.0
