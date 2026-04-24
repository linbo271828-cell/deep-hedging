# Module Architecture — Deep Hedging Lab

This document locks the module-boundary decisions for Deep Hedging Lab.
It supersedes the v0.1 architecture.md for the sprint project.

Last updated: Phase 0 architecture migration (Deep Hedging Lab scaffold).

---

## Guiding Principles

1. Library code lives in `src/`. Experiment entrypoints live in `experiments/`. Never mix them.
2. Each module owns exactly one concern. If two modules need the same logic, it belongs
   in the module with the lower-level concern, and the other imports it.
3. Pure functions are preferred over classes. State (neural weights, optimizer state)
   is acceptable in PyTorch classes only.
4. Transaction cost logic lives in ONE place: `src/hedging/transaction_costs.py`. Period.
5. Risk measure logic lives in ONE place: `src/risk/`. Period.
6. P&L accounting logic lives in ONE place: `src/hedging/pnl.py`. Period.
7. Payoff definitions live in ONE place: `src/payoffs/`. Period.
8. Experiment configs live in ONE place: `src/experiments/configs.py`. Period.
9. Stubs must be typed and raise NotImplementedError — never silently return defaults.

---

## Core Pipeline

```
MarketConfig / HestonConfig
        ↓
src/models/gbm.py or heston.py
  .sample_paths() → np.ndarray (n_paths, n_steps+1)
        ↓
src/payoffs/european.py (or spreads.py, straddles.py)
  payoff functions (model-independent)
  Black-Scholes pricing (GBM-specific)
        ↓
src/hedging/pnl.py
  hedge_pnl(paths, delta_fn, time_grid, ...) → P&L array
        ↓
  ┌───────────────────────────────────────────────────┐
  │ Classical path             Neural path             │
  │ src/hedging/baseline.py    src/neural/train.py     │
  │ (call_delta as delta_fn)   (HedgeNet.forward)      │
  └───────────────────┬───────────────────────────────┘
                      ↓
              src/risk/cvar.py, entropic.py
                      ↓
              src/experiments/summaries.py (SummaryRow)
                      ↓
              src/reporting/ → results/reports/
```

---

## Module Responsibilities (Locked)

### src/models/gbm.py — GBM Path Simulation

Concern: exact-solution GBM path sampling.

Locked public interface:
```python
GBM(s0, mu, sigma)
  .sample_paths(n_paths, n_steps, T, seed) -> np.ndarray  # (n_paths, n_steps+1)
  .log_return_mean(T) -> float
  .log_return_variance(T) -> float

time_grid(n_steps, T) -> np.ndarray  # length n_steps+1

class MarketModel(Protocol)  # satisfied by GBM and Heston
```

Status: COMPLETE. Do not modify without strong reason.
Does NOT own: option pricing, hedging, risk measures.

---

### src/models/heston.py — Heston Stochastic Volatility

Concern: Heston model path simulation (stub).

Locked public interface (when implemented):
```python
HestonParams(v0, kappa, theta, xi, rho)  # dataclass

Heston(s0, r, params)
  .sample_paths(n_paths, n_steps, T, seed) -> np.ndarray  # stock prices
  .sample_paths_with_variance(n_paths, n_steps, T, seed) -> (stock_paths, var_paths)
```

Status: STUB. Implement in Phase C.
Simulation method: Andersen QE scheme (preferred) or Euler-Maruyama (acceptable with notice).

---

### src/payoffs/european.py — European Payoffs + Black-Scholes

Concern: European call/put terminal payoffs AND Black-Scholes closed-form formulas.

Locked public interface:
```python
call_payoff(s_T, k) -> np.ndarray | float
put_payoff(s_T, k)  -> np.ndarray | float

call_price(s, k, r, sigma, tau) -> np.ndarray | float
put_price(s, k, r, sigma, tau)  -> np.ndarray | float
call_delta(s, k, r, sigma, tau) -> np.ndarray | float
put_delta(s, k, r, sigma, tau)  -> np.ndarray | float
call_gamma(s, k, r, sigma, tau) -> np.ndarray | float
call_vega(s, k, r, sigma, tau)  -> np.ndarray | float
call_theta(s, k, r, sigma, tau) -> np.ndarray | float
call_rho(s, k, r, sigma, tau)   -> np.ndarray | float
```

Status: COMPLETE.
Scope note: BS formulas belong here, not in src/models/, because they are
payoff-specific pricing functions, not general model machinery.
Under Heston, pricing uses a different formula (characteristic function).

---

### src/payoffs/spreads.py — Vertical Spread Payoffs

Locked public interface (when implemented):
```python
call_spread_payoff(s_T, k_lo, k_hi) -> np.ndarray | float
put_spread_payoff(s_T, k_lo, k_hi)  -> np.ndarray | float
```

Status: STUB. Implement in Phase B.
No closed-form BS delta for a spread — use component-wise BS deltas in baseline.

---

### src/payoffs/straddles.py — Straddle Payoffs

Locked public interface (when implemented):
```python
straddle_payoff(s_T, k) -> np.ndarray | float
```

Status: STUB. Implement in Phase B.
Classical delta: 2*N(d1) - 1 (combination of call and put deltas).

---

### src/hedging/pnl.py — Canonical P&L Engine

This is the HIGHEST-RISK shared file in the repo.
Both classical and neural paths produce P&L through this function.
Any change to this interface requires lead coordination.

Locked public interface:
```python
hedge_pnl(
    paths: np.ndarray,
    delta_fn: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray],
    time_grid: np.ndarray,
    k: float,
    r: float,
    sigma: float,
    cost_rate: float,
    initial_option_price: float,
) -> np.ndarray  # shape (n_paths,)
```

Accounting model:
- t=0: receive option premium; buy delta_0 shares; pay proportional cost.
- t=i: rebalance to delta_new; pay |trade| * S_i * cost_rate; cash earns exp(r*dt).
- t=T: liquidate position; settle option payoff.

Status: COMPLETE.
Does NOT own: option pricing formulas, risk measures, neural network definition.

---

### src/hedging/transaction_costs.py — Cost Models

Locked public interface:
```python
proportional_cost(trade, s, cost_rate) -> np.ndarray | float
```

Status: COMPLETE.
All cost arithmetic routes through here.
v1: proportional only. Future: add fixed costs, etc. here.

---

### src/hedging/baseline.py — Classical BS Baseline

Locked public interface:
```python
classical_delta_pnl(paths, time_grid, k, r, sigma, cost_rate, initial_option_price) -> np.ndarray
```

Status: COMPLETE.
Thin wrapper calling hedge_pnl with bs.call_delta as delta_fn.
Rule: if no classical baseline exists for a payoff/model, raise NotImplementedError
with a clear explanation. Never silently apply an incorrect baseline.

---

### src/hedging/hedge_universe.py — Hedge Universe (Stub)

Concern: defines which instruments are tradable (Phase D).

Planned protocol:
```python
class HedgeUniverse(Protocol):
    n_instruments: int
    instrument_prices(paths, time_grid) -> np.ndarray  # (n_paths, n_steps+1, n_instruments)

class StockOnlyUniverse     # 1 instrument
class StockPlusOptionUniverse  # 2 instruments: stock + liquid option
```

Status: STUB. Implement in Phase D.

---

### src/risk/cvar.py — Conditional Value-at-Risk

Locked public interface:
```python
cvar(pnl: np.ndarray, alpha: float = 0.95) -> float
```

Status: COMPLETE.
Higher return = MORE risk = WORSE outcome.

---

### src/risk/entropic.py — Entropic Risk Measure

Locked public interface:
```python
entropic_risk(pnl: np.ndarray, lambda_: float = 1.0) -> float
```

Status: COMPLETE.

---

### src/neural/architectures.py — Neural Network Definitions

Locked public interface:
```python
class HedgeNet(nn.Module):
    def __init__(self, n_layers: int = 4, hidden_dim: int = 64) -> None
    def forward(self, x: Tensor) -> Tensor  # x: (batch, n_features), out: (batch,)
```

Status: COMPLETE.
Input features: (S_t/S_0, tau_t, prev_delta) — dimension 3.
Output: sigmoid-bounded hedge ratio in (0, 1).
Does NOT own: training loop, features, P&L accounting.

---

### src/neural/features.py — Feature Builders

Locked public interface:
```python
class FeatureBuilder(Protocol): ...  # callable protocol

base_features(s_t, s0, tau_t, prev_delta) -> np.ndarray  # (n_paths, 3)
heston_features(s_t, s0, tau_t, prev_delta, v_t) -> np.ndarray  # (n_paths, 4)  STUB
```

Status: base_features COMPLETE; heston_features STUB.

---

### src/neural/train.py — Training Loop

Locked public interface:
```python
train(
    config: TrainingConfig,
    market_config: MarketConfig,
    model: HedgeNet,
    optimizer: torch.optim.Optimizer,
) -> tuple[HedgeNet, list[float]]
```

Internal `_compute_pnl` is a differentiable PyTorch mirror of `src/hedging/pnl.hedge_pnl`.
The two must stay in sync: if pnl.py accounting changes, _compute_pnl must change too.

Status: COMPLETE.

---

### src/neural/evaluate.py — Evaluation

Planned public interface:
```python
@dataclass
class EvaluationResult: ...  # all standard metrics

evaluate(model, market_config, training_config, eval_seed) -> EvaluationResult
```

Status: STUB. Implement in Phase A.
Separation from training loop is mandatory: training loss != evaluation metrics.

---

### src/experiments/configs.py — Experiment Hyperparameters

HIGH-RISK SHARED FILE. Rules: no field removal, no field rename without updating all callers.

Frozen dataclasses:
```python
MarketConfig          # GBM parameters
HestonConfig          # Heston parameters (stub)
PayoffConfig          # payoff_type, strike(s)
HedgeUniverseConfig   # universe_type, hedge option params
TrainingConfig        # n_paths, n_epochs, lr, seed, cost_rate, alpha
EvaluationConfig      # eval_n_paths, seed_offset, alpha, lambda_
ExperimentConfig      # umbrella: name + all sub-configs
```

Status: COMPLETE.

---

### src/experiments/registry.py — Named Experiment Registry

Public interface:
```python
REGISTRY: dict[str, ExperimentConfig]
get_config(name: str) -> ExperimentConfig
list_experiments() -> list[str]
```

Status: PARTIAL. Phase A baseline configs registered.

---

### src/experiments/runner.py — Experiment Runner (Stub)

Public interface:
```python
run_experiment(name: str, force: bool = False) -> list[SummaryRow]
```

Status: STUB. Implement in Phase A.

---

### src/experiments/summaries.py — SummaryRow

```python
@dataclass
class SummaryRow: ...  # canonical result row for a (config, strategy, seed) triple
```

Status: COMPLETE. All lanes write SummaryRow. Do not add new metrics without checking
that all callers can supply them.

---

### src/reporting/ — Report Generation (Stubs)

Three stubs:
- `tables.py`: build_summary_table(rows) -> pd.DataFrame
- `plots.py`: plot_cost_frontier(rows, output_path), ...
- `report.py`: generate_report(results_dir, output_path)

Status: STUB. Implement in Phase E.

---

### src/utils/ — Shared Utilities

```python
seeds.py: derive_seed(base_seed, purpose) -> int
io.py: save_json, load_json, save_dataclass_json
stats.py: standard_error(arr), report_moment(arr, name)
```

Status: COMPLETE (seeds, stats, basic io).

---

### experiments/ — Named Experiment Entrypoints

v0.1 baseline scripts (01-05) are preserved as-is.
New scripts use `src/experiments/registry.get_config(name)` to resolve configs.
Each script saves to `results/raw/<name>/` via `src/utils/io`.

---

### tests/ — Numerical Correctness Tests

Philosophy: test mathematical claims, not code structure.
No mocking. Real numerics. Total suite < 60 seconds.

| Test file | Tests |
|---|---|
| test_market.py | 7 GBM tests |
| test_black_scholes.py | 7 BS tests |
| test_hedger.py | 5 P&L accounting tests |
| test_risk_measures.py | 15 risk measure tests |

New tests needed (as implementation progresses):
- test_payoffs.py: spread payoffs, straddle payoffs
- test_heston.py: Heston path properties
- test_neural_smoke.py: forward pass shape, training loss decreases
- test_runner_smoke.py: runner returns SummaryRows

---

## Data Flow: Seed Management

```
ExperimentConfig.training.seed  (master seed)
        ↓
derive_seed(seed, "train")  → GBM training paths
derive_seed(seed, "eval")   → GBM evaluation paths
derive_seed(seed, "val")    → intermediate validation (optional)

torch.manual_seed(seed)  called once at top of train()
```

Never: np.random.seed(), torch.manual_seed() globally.
Always: pass seeds explicitly through configs.

---

## Shared / High-Risk Files

| File | Risk | Why |
|---|---|---|
| src/experiments/configs.py | VERY HIGH | Every experiment and train() imports it |
| src/hedging/pnl.py | VERY HIGH | Both classical and neural paths depend on it |
| src/neural/train.py | HIGH | _compute_pnl must mirror pnl.py exactly |
| src/experiments/registry.py | HIGH | Named configs drive all experiments |
| src/payoffs/european.py | HIGH | BS signatures used everywhere |
| src/risk/cvar.py | MEDIUM | Training loss + evaluation metric |
| pyproject.toml | MEDIUM | Affects all lint/typecheck/test runs |

---

## How to Map v0.1 to v1

| v0.1 flat module | v1 canonical location |
|---|---|
| src/market.py | src/models/gbm.py |
| src/black_scholes.py | src/payoffs/european.py |
| src/hedger.py (hedge_pnl) | src/hedging/pnl.py |
| src/hedger.py (classical_delta_pnl) | src/hedging/baseline.py |
| src/risk_measures.py (cvar) | src/risk/cvar.py |
| src/risk_measures.py (entropic_risk) | src/risk/entropic.py |
| src/neural_hedger.py | src/neural/architectures.py |
| src/train.py | src/neural/train.py |
| src/config.py | src/experiments/configs.py |
