# Module Architecture — Deep Hedging Sprint

This document locks the module-boundary decisions for this project.
It is the authoritative reference when there is any doubt about where logic belongs.

Last updated: initial architecture lock (Days 1-2 complete).

---

## Guiding Principles

1. Library code belongs in `src/`. Experiment scripts belong in `experiments/`. Never mix them.
2. Each module owns exactly one concern. If two modules need the same logic, it belongs in the
   module with the lower-level concern, and the other imports it.
3. Pure functions are preferred over classes. State (neural network weights, optimizer state)
   is acceptable in PyTorch classes only.
4. There are no "utility" catch-all modules. Every file has a clearly named responsibility.
5. Transaction cost logic lives in ONE place: `src/hedger.py`. Period.
6. Risk measure logic lives in ONE place: `src/risk_measures.py`. Period.

---

## Module Responsibilities (Locked)

### src/market.py — Path Generation Only

Concern: generating Monte Carlo asset price paths.

Public interface:
```
GBM(s0, mu, sigma)
  .sample_paths(n_paths, n_steps, T, seed) -> ndarray shape (n_paths, n_steps+1)
  .log_return_mean(T) -> float
  .log_return_variance(T) -> float

time_grid(n_steps, T) -> ndarray length (n_steps+1)

class MarketModel(Protocol): # for future model extensions (Heston, etc.)
```

Does NOT own: option pricing, hedge deltas, cost accounting, risk measures.
Status: COMPLETE. Do not modify without strong reason.

---

### src/black_scholes.py — Closed-Form Pricing and Greeks

Concern: Black-Scholes analytical formulas.

Public interface:
```
call_price(s, k, r, sigma, tau) -> ndarray | float
put_price(s, k, r, sigma, tau) -> ndarray | float
call_delta(s, k, r, sigma, tau) -> ndarray | float
call_gamma(s, k, r, sigma, tau) -> ndarray | float
call_vega(s, k, r, sigma, tau) -> ndarray | float
call_theta(s, k, r, sigma, tau) -> ndarray | float
call_rho(s, k, r, sigma, tau) -> ndarray | float
```

All functions are pure: no side effects, no state, no randomness.
Handles tau=0 via np.where (intrinsic value, not NaN or ZeroDivision).
Status: COMPLETE. Signatures locked — do not change without updating all callers.

---

### src/hedger.py — P&L Accounting and Classical Hedger

Concern: computing hedge portfolio P&L across time, including transaction costs.

This module is the shared P&L engine. Both the classical delta hedger and the neural hedger
produce P&L by calling functions from this module. This is the highest-risk shared file.

Planned public interface:
```
hedge_pnl(
    paths: np.ndarray,           # shape (n_paths, n_steps+1)
    delta_fn: callable,          # callable(s, tau, k, r, sigma) -> delta, same shape as s
    time_grid: np.ndarray,       # shape (n_steps+1,)
    k: float,
    r: float,
    sigma: float,
    cost_rate: float,            # proportional transaction cost in [0, 1]; 0.0005 = 5 bps
    initial_option_price: float, # premium received for selling the call
) -> np.ndarray                  # shape (n_paths,), terminal P&L per path

classical_delta_pnl(
    paths: np.ndarray,
    time_grid: np.ndarray,
    k: float, r: float, sigma: float,
    cost_rate: float,
    initial_option_price: float,
) -> np.ndarray                  # thin wrapper calling hedge_pnl with bs.call_delta
```

P&L accounting logic (what it does):
- At t=0: receive option premium, set initial delta position
- At each rebalancing step: observe new delta signal, compute trade = new_delta - old_delta,
  pay |trade| * S_t * cost_rate in transaction costs
- At T: pay out the call payoff max(S_T - K, 0), collect the hedge portfolio value

Transaction cost logic lives ONLY here.
Neural hedger training also routes P&L through these functions.

---

### src/risk_measures.py — Risk Aggregation (Pure Functions)

Concern: mapping a distribution of P&L outcomes to a scalar risk measure.

Planned public interface:
```
cvar(pnl: np.ndarray, alpha: float = 0.95) -> float
    # CVaR (Expected Shortfall): mean of the worst (1-alpha) fraction of outcomes
    # pnl shape: (n_paths,). Returns a positive number (larger = worse).
    # Convention: CVaR measures LOSS, so negate P&L internally.

entropic_risk(pnl: np.ndarray, lambda_: float = 1.0) -> float
    # Entropic risk measure: (1/lambda) * log(E[exp(-lambda * pnl)])
    # Convex, monotone, consistent risk measure.
```

All functions are pure: numpy in, float out. No model dependencies, no side effects.
These functions are used as:
  (a) training loss in src/train.py
  (b) evaluation metric in experiment scripts

The sign convention is: higher return value = MORE risk = WORSE outcome.
Internally, losses are positive (negate P&L before computing CVaR/entropic).

---

### src/neural_hedger.py — PyTorch Model Definition Only

Concern: defining the neural network architecture.

Planned public interface:
```
class HedgeNet(nn.Module):
    def __init__(self, n_layers: int = 4, hidden_dim: int = 64) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
        # x shape: (batch, 3) — features: (S_t/S_0, tau_t, current_delta)
        # output shape: (batch,) — hedge ratio in [0, 1] via sigmoid
```

Does NOT own: training loop, loss computation, P&L accounting, data generation.
Does NOT call numpy. PyTorch only in this file.

Design decisions:
- Sigmoid output bounds the hedge ratio to [0, 1]. A call delta is always in [0, 1], so this
  is correct and avoids exploding positions.
- Input normalization: S_t/S_0 (not raw price), tau_t in [0, T] (natural scale).
  current_delta is already in [0, 1].
- Architecture: FFN with ReLU activations and layer normalization for training stability.
  Default: 4 layers, 64 hidden units. Adjust in TrainingConfig.

---

### src/train.py — Training Loop

Concern: training the neural hedger.

Planned public interface:
```
train(
    config: TrainingConfig,
    market_config: MarketConfig,
    model: HedgeNet,
    optimizer: torch.optim.Optimizer,
) -> tuple[HedgeNet, list[float]]
    # returns trained model and list of per-epoch loss values
```

Internals:
- Calls `GBM.sample_paths()` with `config.seed` to generate training paths
- Converts paths to tensors, computes features at each timestep
- Calls `model.forward()` at each step to get hedge ratios
- Routes through `hedge_pnl()` for P&L accounting (torch-differentiable version)
- Calls `cvar()` as the loss
- Backpropagates and steps the optimizer
- Saves checkpoint at end of training

Note: `hedge_pnl` in train.py needs a differentiable PyTorch version. This is implemented
locally inside train.py using torch tensors, NOT by calling the numpy version in hedger.py.
The numpy `hedge_pnl` in hedger.py is for evaluation only.

---

### src/config.py — Experiment Hyperparameters

Concern: all tunable parameters in one place.

Planned public interface:
```
@dataclass(frozen=True)
class MarketConfig:
    s0: float = 100.0
    mu: float = 0.05         # physical drift (not used in risk-neutral training)
    r: float = 0.02          # risk-free rate
    sigma: float = 0.20
    T: float = 0.5           # maturity in years
    n_steps: int = 50        # rebalancing steps

@dataclass(frozen=True)
class TrainingConfig:
    n_paths: int = 50_000    # paths per training epoch
    n_epochs: int = 200
    lr: float = 1e-3
    hidden_dim: int = 64
    n_layers: int = 4
    seed: int = 42
    cost_rate: float = 0.0   # 0.0 = no costs; 0.0005 = 5 bps
    alpha: float = 0.95      # CVaR level
    k: float = 100.0         # strike price
```

This is a high-risk shared file: changes here affect every experiment.
Do not remove fields. Do not rename fields without updating all callers.
Add new fields with defaults to avoid breaking existing code.

---

### experiments/ — Self-Contained Experiment Scripts

Each script:
- Has `if __name__ == "__main__":` guard
- Instantiates configs from `src/config.py`
- Does its work
- Saves plots to `plots/`
- Prints a brief numerical summary to stdout
- Takes no command-line arguments

Planned experiments:

| Script | What it does | Key output |
|---|---|---|
| 01_validate_market.py | GBM empirical moments vs. theoretical | plots/01_gbm_moments.png |
| 02_classical_delta.py | Delta hedger P&L, no cost + 5 bps cost | plots/02_pnl_distribution.png |
| 03_neural_no_costs.py | Train neural hedger, show it learns BS delta | plots/03_learned_vs_bs_delta.png |
| 04_neural_with_costs.py | Retrain with 5 bps cost, show deviation from BS | plots/04_hedge_ratio_with_costs.png |
| 05_cost_frontier.py | Sweep costs 0→50 bps, CVaR vs. expected cost | plots/05_cost_frontier.png |

---

### tests/ — Numerical Correctness Tests

One test file per `src/` module that has testable numerical behavior.
Tests are fast (total suite < 60s), seeded, deterministic.
Tests verify mathematical claims, not internal implementation structure.

| Test file | Tests |
|---|---|
| test_market.py | 7 tests — shape, determinism, positivity, moments, input validation |
| test_black_scholes.py | 7 tests — put-call parity, FD Greeks, expiry, limits, MC agreement |
| test_hedger.py | to be written — P&L zero-mean in no-cost limit, cost deduction accuracy |
| test_risk_measures.py | to be written — CVaR on known dist, entropic risk convexity |

---

### writeup/ — Documentation

| File | Owner | Content |
|---|---|---|
| math_reference.md | User | BS PDE derivation, Ito's lemma, Greeks, risk measures |
| paper_notes.md | User | Buehler 2019 annotation in own words |
| architecture.md | Claude | THIS FILE — locked module boundaries |
| sprint_plan.md | Claude | Build order, gates, status |

---

## Data Flow Diagram

```
                    src/config.py
                   (seeds, params)
                        |
           +------------+------------+
           |                         |
    src/market.py              src/black_scholes.py
    GBM.sample_paths()         call_delta(), call_price()
           |                         |
           +------------+------------+
                        |
                 src/hedger.py
                 hedge_pnl()
                        |
               +---------+---------+
               |                   |
      Classical path          Neural path
      (call_delta as         (HedgeNet.forward()
       delta_fn)              via src/train.py)
               |                   |
               +----+----+---------+
                    |
           src/risk_measures.py
           cvar(), entropic_risk()
                    |
              experiments/
              plots + prints
```

---

## How Seeds Are Passed

- GBM paths: `gbm.sample_paths(..., seed=config.seed)`. Always explicit.
- Training paths use `config.seed`. Validation/test paths use `config.seed + 1000`.
  (This ensures train/val paths are independent.)
- PyTorch: `torch.manual_seed(config.seed)` called once at the top of `train()`.
- Never call `np.random.seed()` or set any global RNG state.
- Never share a seed between two experiments that are meant to be independent.

---

## How Outputs Are Saved

- Plots: `plt.savefig(f"plots/{filename}.png", dpi=150, bbox_inches="tight")`
- Model checkpoints: `torch.save(model.state_dict(), f"plots/{name}_checkpoint.pt")`
  (checkpoint stored alongside plots for now; move to `checkpoints/` if needed)
- Numerical results: printed to stdout by experiment scripts. Redirect to log if needed.
- No results database. No CSV dumps. No pandas DataFrames written to disk. Keep it simple.

---

## Agent-Team Ownership Boundaries

### Lane A — Quant Core
Owns: `src/market.py`, `src/black_scholes.py`, `src/hedger.py`, `tests/test_hedger.py`
Reads (do not write): `src/config.py`
Safe changes: fixing numerical edge cases, adding Greeks, improving P&L accounting efficiency
Needs coordination: changing any public function signatures in market.py or black_scholes.py
  (callers are in experiments/, train.py, and tests/)

### Lane B — Neural Modeling
Owns: `src/neural_hedger.py`, `src/train.py`
Reads (do not write): `src/hedger.py`, `src/risk_measures.py`, `src/config.py`
Safe changes: model architecture, training loop, optimizer choice, learning rate schedule
Needs coordination: any change to hedge_pnl interface (Lane A owns it);
  any change to cvar/entropic_risk interface (Lane C owns it);
  any new config fields (Lane C owns config.py)

### Lane C — Risk and Config
Owns: `src/risk_measures.py`, `src/config.py`, `tests/test_risk_measures.py`
Reads (do not write): nothing in src/ (it is a low-level dependency)
Safe changes: adding new risk measures, adjusting default hyperparams, adding new config fields
Needs coordination: removing or renaming existing config fields (breaks Lanes A, B, D);
  changing risk measure signatures (breaks Lane B training loop)

### Lane D — Experiments and Plots
Owns: `experiments/`, `plots/`
Reads (do not write): all of `src/`
Safe changes: writing new experiment scripts, adjusting plot aesthetics, adding analysis
Needs coordination: any new config fields needed (request from Lane C);
  any new src/ functions needed (request from Lane A or B)

### Lane E — Docs and Writeup
Owns: `writeup/`, `README.md`, `CLAUDE.md`
Reads: nothing that affects behavior
Safe changes: all documentation, README sections, status tables, math explanations
Needs coordination: nothing — docs are fully isolated from code behavior

---

## Shared / High-Risk Files

| File | Risk | Why |
|---|---|---|
| src/config.py | HIGH | Every experiment and train.py imports it; field renames break everything |
| src/hedger.py | HIGH | Called from both classical and neural paths; signature change breaks both |
| src/risk_measures.py | MEDIUM | Called from train.py and experiment scripts |
| pyproject.toml | MEDIUM | Tool config affects all lint/typecheck/test runs |
| Makefile | LOW | Only affects `make` targets |

When editing high-risk files, check all callers before changing any signature.
