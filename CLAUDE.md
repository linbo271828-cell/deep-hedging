# CLAUDE.md — Deep Hedging Sprint

This file is the primary context document for any Claude Code session working on this repo.
Read it fully before touching any file. It describes what exists, what the rules are, and
what to do next.

---

## Project Mission

Implement a neural-network-based option hedger in the style of Buehler, Gonon, Teichmann,
and Wood (2019), benchmarked against classical Black-Scholes delta-hedging, on a European
call option under proportional transaction costs, using simulated GBM markets.

This is a **quant-finance portfolio project** for quant trading / quant research internships
(Jane Street, Citadel, Two Sigma, Optiver, Hudson River, DRW). It doubles as a strong SWE
portfolio piece because of its clean architecture, type discipline, and reproducibility.

The finished repo must read as: **math + modeling + disciplined engineering**.
It must NOT read as: hype deep-learning repo, generic research platform, or overengineered framework.

---

## Portfolio / Resume Positioning

What this demonstrates:
- Ability to read a math-heavy paper and implement it faithfully
- Working knowledge of Black-Scholes, GBM, Greeks, risk-neutral pricing
- Awareness that market frictions matter (transaction costs, not just idealized pricing)
- Honest statistical evaluation (P&L distributions, CVaR, not cherry-picked means)
- Clean, typed, reproducible, tested Python that can land PRs without supervision

What this does NOT claim:
- Full reproduction of Buehler (2019) — this is a simplified, sprint-scoped subset
- Real-market calibration — everything is simulated GBM
- Novel research contribution — it is a careful implementation of an existing idea
- HFT applicability — this is batch offline learning, not latency-sensitive

The README must state the above limitations honestly. Overclaiming destroys quant credibility.

---

## MVP Scope (This Sprint)

In scope:
- GBM path simulator (exact solution, no Euler-Maruyama)
- Black-Scholes closed-form pricer + all Greeks
- Classical delta hedger — no costs and with proportional transaction costs
- CVaR and entropic risk measures
- Neural hedger: feedforward network (PyTorch), inputs (S_t/S_0, tau_t, current_delta)
- Training loop minimizing CVaR of terminal P&L including transaction costs
- Zero-cost sanity check: neural hedger must recover BS delta before any cost experiments
- Cost sweep 0→50 bps: headline cost-frontier plot
- Reproducible experiments, seeded, Makefile-driven

Explicitly out of scope for this sprint:
- Heston / stochastic volatility (stretch only if days 1-10 finish early)
- Recurrent architectures (LSTM, attention)
- Real market data / calibration
- Exotic or path-dependent options
- Multi-asset hedging
- Distributed training, Docker, cloud infra, web backends
- Any UI or visualization beyond matplotlib static plots
- Continuous-time limit analysis

Do NOT add features beyond this scope without explicit user instruction.

---

## Current Repository Status

As of the last update to this file:

| File | Status |
|---|---|
| src/market.py | COMPLETE — GBM simulator, tested |
| src/black_scholes.py | COMPLETE — BS pricer + Greeks, tested |
| tests/test_market.py | COMPLETE — 7 tests, all passing |
| tests/test_black_scholes.py | COMPLETE — 7 tests, all passing |
| src/__init__.py | COMPLETE |
| tests/__init__.py | COMPLETE |
| pyproject.toml | COMPLETE |
| requirements.txt | COMPLETE |
| Makefile | COMPLETE |
| .gitignore | COMPLETE |
| src/hedger.py | COMPLETE — hedge_pnl + classical_delta_pnl, 5 tests passing |
| src/risk_measures.py | COMPLETE — cvar + entropic_risk, 15 tests passing |
| src/neural_hedger.py | COMPLETE — HedgeNet FFN (sigmoid output, 3 features) |
| src/train.py | COMPLETE — CVaR training loop, differentiable torch P&L |
| src/config.py | COMPLETE — MarketConfig + TrainingConfig frozen dataclasses |
| tests/test_hedger.py | COMPLETE — 5 tests, all passing |
| tests/test_risk_measures.py | COMPLETE — 15 tests, all passing |
| experiments/01_validate_market.py | COMPLETE — GBM moment validation, plot saved |
| experiments/02_classical_delta.py | COMPLETE — classical delta P&L, plot saved |
| experiments/03_neural_no_costs.py | COMPLETE — gate passed MAD=0.035 < 0.05 |
| experiments/04_neural_with_costs.py | COMPLETE — 5 bps cost experiment, plot saved |
| experiments/05_cost_frontier.py | COMPLETE — headline figure generated |
| writeup/math_reference.md | SCAFFOLDED — user fills in |
| writeup/paper_notes.md | SCAFFOLDED — user fills in |
| writeup/architecture.md | COMPLETE |
| writeup/sprint_plan.md | COMPLETE |
| README.md | COMPLETE — results and limitations documented |
| CLAUDE.md | THIS FILE |

Update this table whenever a file changes status.

---

## Directory Map

```
deep-hedging/
├── CLAUDE.md                       # This file. Read first.
├── README.md                       # Public-facing writeup for GitHub
├── pyproject.toml                  # Project config + tool settings
├── requirements.txt                # Flat dep list for venv install
├── Makefile                        # make test / lint / reproduce / clean
├── .gitignore
│
├── src/                            # All importable library code
│   ├── __init__.py
│   ├── market.py                   # GBM simulator (COMPLETE)
│   ├── black_scholes.py            # BS pricer + Greeks (COMPLETE)
│   ├── hedger.py                   # P&L accounting, classical delta hedger
│   ├── risk_measures.py            # CVaR, entropic risk (pure functions)
│   ├── neural_hedger.py            # PyTorch FFN hedge-ratio network
│   ├── train.py                    # Training loop (uses GBM + neural_hedger + risk)
│   └── config.py                   # Experiment hyperparams as frozen dataclasses
│
├── experiments/                    # One script per experiment. Imports src/. Saves to plots/.
│   ├── 01_validate_market.py       # GBM moment validation plot
│   ├── 02_classical_delta.py       # Classical delta hedger P&L (no cost + with cost)
│   ├── 03_neural_no_costs.py       # Train neural hedger, verify it learns BS delta
│   ├── 04_neural_with_costs.py     # Retrain with transaction costs
│   └── 05_cost_frontier.py         # Sweep costs, produce headline figure
│
├── tests/                          # pytest — numerical correctness only
│   ├── __init__.py
│   ├── test_market.py              # 7 GBM tests (COMPLETE)
│   ├── test_black_scholes.py       # 7 BS tests (COMPLETE)
│   ├── test_hedger.py              # to be written alongside hedger.py
│   └── test_risk_measures.py       # to be written alongside risk_measures.py
│
├── plots/                          # Generated figures. gitignored (except .gitkeep)
│   └── .gitkeep
│
├── notebooks/                      # Scratch / derivations only. Not production code.
│   └── 00_derivations.ipynb        # (optional) hand derivations typeset
│
└── writeup/                        # Documentation + math
    ├── math_reference.md           # BS PDE derivation, Greeks, risk measures
    ├── paper_notes.md              # Buehler 2019 annotations in own words
    ├── architecture.md             # Module-boundary decisions (locked)
    └── sprint_plan.md              # Concrete build order for remaining work
```

---

## Module Responsibilities

### src/market.py — COMPLETE, DO NOT MODIFY WITHOUT GOOD REASON

Owns: GBM path generation only.
Protocol `MarketModel` defines the interface for future model extensions (Heston).
`GBM.sample_paths()` returns shape `(n_paths, n_steps + 1)`. Index 0 is `s0`.
`time_grid()` returns a uniform grid of length `n_steps + 1`.

Does NOT own: option pricing, hedging, costs, risk measures.

Contract: `sample_paths` is deterministic given `seed`. Never break this.

---

### src/black_scholes.py — COMPLETE, DO NOT MODIFY WITHOUT GOOD REASON

Owns: closed-form BS pricing for European calls/puts, all Greeks (delta, gamma, vega, theta, rho).
All functions are pure: no state, no side effects, no simulation.
Handles `tau=0` edge case cleanly via `np.where` (intrinsic value, not NaN).

Does NOT own: simulation, hedging, risk, training.

Contract: `call_price`, `call_delta` signatures must remain stable — many things import them.

---

### src/hedger.py — TO BUILD

Owns:
- `hedge_pnl(paths, delta_fn, cost_rate, initial_option_price)` → array of shape `(n_paths,)`
  This is the P&L accounting engine. Both the classical hedger and the neural hedger share it.
- `classical_delta_hedger(paths, time_grid, k, r, sigma, cost_rate)` → P&L array
  Thin wrapper that calls hedge_pnl with `delta_fn = bs.call_delta`.

Transaction cost logic lives HERE, not in neural_hedger or train.py.
P&L accounting logic lives HERE, not scattered across experiments.

Does NOT own: option pricing formulas, risk measure aggregation, neural network definition.

---

### src/risk_measures.py — TO BUILD

Owns: pure functions that take a 1D array of P&L outcomes and return a scalar risk value.
- `cvar(pnl, alpha=0.95)` → float  (expected shortfall at level alpha)
- `entropic_risk(pnl, lambda_=1.0)` → float

All functions are pure: no model dependencies, no side effects.
These become the training loss functions AND the evaluation metrics.

Does NOT own: path generation, hedging logic, training.

---

### src/neural_hedger.py — TO BUILD

Owns: the PyTorch neural network definition only.
- `HedgeNet(n_layers, hidden_dim)` — FFN, maps features → hedge ratio in [0, 1]
- Input features: `(S_t / S_0, tau_t, current_delta)` — normalized, dimension 3
- Output: scalar hedge ratio (sigmoid-bounded or tanh-shifted to [0,1])

Does NOT own: training loop, loss computation, data generation, P&L accounting.

---

### src/train.py — TO BUILD

Owns: the training loop.
- `train(config, model, optimizer, seed)` → trained model + loss history
- Generates GBM paths inside the training loop (seeded)
- Calls `hedger.hedge_pnl()` for P&L accounting
- Calls `risk_measures.cvar()` as the loss
- Saves final model checkpoint to a path specified in config

Does NOT own: model architecture, risk measure formulas, plotting, experiment logic.

---

### src/config.py — TO BUILD

Owns: experiment hyperparameters as frozen dataclasses.
- `MarketConfig(s0, mu, r, sigma, T, n_steps)`
- `TrainingConfig(n_paths, n_epochs, lr, hidden_dim, n_layers, seed, cost_rate, alpha)`

Seeds live here. All experiments instantiate one of these configs.
This is a high-risk shared file — changes here can break multiple experiments.

Does NOT own: any logic. Data containers only.

---

### experiments/ — TO BUILD

Each experiment script is self-contained:
- Imports from `src/` only
- Instantiates a config from `src/config.py`
- Produces one or more plots saved to `plots/`
- Prints a short numerical summary to stdout
- Takes no command-line arguments (everything is in the config)

Experiments do NOT contain library logic. If logic appears in two experiments, it belongs in `src/`.

---

### tests/ — PARTIALLY COMPLETE

Philosophy: tests verify numerical claims, not implementation details.
Do not test private functions. Test observable behavior against closed-form expectations.
Every test must pass in under 30 seconds total.
Monte Carlo tests use at most 200k paths and a fixed seed.

New tests required:
- `test_hedger.py`: verify P&L is zero-mean in no-cost / no-discretization limit;
  verify transaction costs reduce expected P&L by the expected amount.
- `test_risk_measures.py`: verify CVaR on known distributions (e.g., uniform, Gaussian)
  matches closed-form; verify entropic risk is convex.

---

## Coding Standards

- Python 3.11 type annotations on all public functions. Use `np.ndarray` not `NDArray`.
- Line length: 100 (ruff + black configured in pyproject.toml).
- ruff rules: E, F, I, N, UP, B, SIM, RUF. Run `make lint` before committing.
- mypy strict mode. All `src/` must type-check clean. Tests are excluded from strict mypy.
- Docstrings on all public functions. Format: one-line summary, then math in plain text.
- No `print()` in `src/` except for debug guards (`if __debug__:`). Use stdout only in experiments.
- No global mutable state. No global random state.
- Prefer explicit over implicit. Pass seeds explicitly. Pass configs explicitly.
- No `import *`.
- Do not add dependencies beyond what is in `requirements.txt` without explicit approval.

---

## Numerical Rigor Expectations

- GBM paths use the exact exponential solution. No Euler-Maruyama. This is non-negotiable.
- Black-Scholes formulas must match finite-difference Greeks to the tolerances in the tests.
- Monte Carlo results must agree with closed-form within 3 standard errors at 200k paths.
- Any time you compute a moment empirically, also compute the standard error and report it.
- Never use `np.random.seed()` (global state). Always use `np.random.default_rng(seed)`.
- Never use `torch.manual_seed()` globally. Set it locally inside `train()`.
- All float comparisons in tests use `np.testing.assert_allclose` with explicit `atol`/`rtol`.
  Never use bare `==` on floats.

---

## Testing Philosophy

- Tests live in `tests/`. Experiments live in `experiments/`. Do not mix them.
- Tests must be fast: total suite under 60 seconds.
- Tests verify mathematical claims, not code structure. If a formula is wrong, the test catches it.
- Do not mock numerical functions. Test them for real with appropriate sample sizes.
- A test that always passes regardless of the implementation is worthless.
- The zero-cost neural hedger recovering Black-Scholes delta is NOT a unit test — it is
  an experiment gate (see sprint plan). Do not add it to `tests/`.

---

## Reproducibility Rules

- Every experiment is reproducible from seed alone. No randomness outside of seeded RNG.
- `make reproduce` must regenerate every plot from scratch without manual intervention.
- Seeds are stored in `src/config.py` dataclasses, not hardcoded in experiment scripts.
- Model checkpoints are saved to `plots/` or a `checkpoints/` directory (TBD), not committed.
- Numerical summaries are printed to stdout by experiment scripts and can be piped to a log.
- Never rely on the order of dict iteration (use sorted keys when order matters for reproducibility).

---

## Seed Discipline Rules

- NumPy: `rng = np.random.default_rng(config.seed)`. Pass `rng` explicitly. Do NOT call `rng` again
  after the first call in a function — pass sub-seeds derived from it if needed.
- PyTorch: `torch.manual_seed(config.seed)` called at the top of `train()`. Nowhere else.
- GBM: `gbm.sample_paths(..., seed=config.seed)` — always explicit.
- Different experiments use different seeds (defined in config). Never re-use seeds across
  experiments that are meant to be independent.
- Validation paths and training paths must use different seeds.

---

## Experiment Conventions

- Experiment scripts are named `{NN}_{short_descriptor}.py` (e.g., `03_neural_no_costs.py`).
- Each script prints a 3-5 line summary to stdout: key numbers, no clutter.
- Each script saves exactly the plots it generates, named `{NN}_{descriptor}.png`.
- Scripts have a `if __name__ == "__main__":` guard.
- No argparse. Configs are in `src/config.py`. Modify config there, not via CLI flags.
- Scripts should run in under 5 minutes on a laptop CPU (adjust n_paths / n_epochs accordingly).

---

## Plotting / Output Conventions

- All plots saved to `plots/` as PNG, 150 dpi minimum.
- Figure size: (8, 5) for single-panel, (12, 5) for side-by-side.
- Always label axes. Always add a title. Always add a legend if more than one series.
- Color palette: matplotlib default is fine. Do not add seaborn unless it saves significant work.
- Plots are gitignored (`.gitignore` already excludes `plots/*.png`). Only the headline figure
  (`plots/05_cost_frontier.png`) should be manually committed for the README.
- The headline figure (`05_cost_frontier.png`) is the most important output of the project.
  It must show the Pareto frontier of risk vs. expected cost for both strategies.

---

## Documentation Conventions

- `writeup/math_reference.md`: user's own derivations and math notes. Written by the user, not by
  Claude. Claude provides structure/scaffold only.
- `writeup/paper_notes.md`: user's annotation of Buehler 2019. Same ownership rule.
- `writeup/architecture.md`: module-boundary decisions. Written by Claude. Locked.
- `writeup/sprint_plan.md`: build order and sprint status. Updated as tasks complete.
- `README.md`: public-facing. Written collaboratively. Must be honest about scope.
- `CLAUDE.md`: this file. Updated whenever repo status changes significantly.

---

## Naming Conventions

- Functions: `snake_case`. Classes: `PascalCase`. Constants: `UPPER_SNAKE`.
- P&L variable: always `pnl` (not `profit`, `return`, `gain`).
- Hedge ratio / delta: always `delta` (not `hedge`, `position`, `h`).
- Time to maturity: always `tau` (not `ttm`, `time_left`, `T_minus_t`).
- Number of paths: always `n_paths`. Number of steps: always `n_steps`.
- Cost rate: always `cost_rate` (not `kappa`, `tc`, `spread`).
- Risk level for CVaR: always `alpha` (not `confidence`, `level`).
- Model configs: `MarketConfig`, `TrainingConfig` (from `src/config.py`).

---

## File Ownership Boundaries (Agent-Team Reference)

### Lane A — Quant Core
Primary: `src/market.py`, `src/black_scholes.py`, `src/hedger.py`, `tests/test_hedger.py`
Shared (read only): `src/config.py`
Safe changes: fixing formulas, adding Greeks, improving P&L accounting
Coordination required before: changing function signatures in market.py or black_scholes.py

### Lane B — Neural Modeling
Primary: `src/neural_hedger.py`, `src/train.py`
Shared (read only): `src/hedger.py`, `src/risk_measures.py`, `src/config.py`
Safe changes: model architecture, training loop logic, optimizer, learning rate schedule
Coordination required before: changing the P&L accounting interface in hedger.py,
changing risk_measures signatures

### Lane C — Risk & Config
Primary: `src/risk_measures.py`, `src/config.py`, `tests/test_risk_measures.py`
Shared (read only): nothing
Safe changes: adding risk measures, adjusting hyperparams, adding config fields
Coordination required before: removing config fields (breaks everyone), renaming fields

### Lane D — Experiments & Plots
Primary: `experiments/`, `plots/`
Shared (read only): all of `src/`, `src/config.py`
Safe changes: writing new experiment scripts, adjusting plots, adding analysis
Coordination required before: adding new config fields (coordinate with Lane C)

### Lane E — Docs & Writeup
Primary: `writeup/`, `README.md`, `CLAUDE.md`
Shared: none
Safe changes: all writeup content, README sections, CLAUDE.md status table
Coordination required before: nothing (docs are fully isolated)

---

## Rules About Not Breaking Module Contracts

These function signatures are locked. Do not change them without updating ALL callers:

```
# src/market.py
GBM.sample_paths(n_paths: int, n_steps: int, T: float, seed: int) -> np.ndarray
# returns shape (n_paths, n_steps + 1)

# src/black_scholes.py
call_price(s, k, r, sigma, tau) -> np.ndarray | float
call_delta(s, k, r, sigma, tau) -> np.ndarray | float

# src/risk_measures.py (once built)
cvar(pnl: np.ndarray, alpha: float = 0.95) -> float
entropic_risk(pnl: np.ndarray, lambda_: float = 1.0) -> float

# src/hedger.py (once built)
hedge_pnl(paths, delta_fn, cost_rate, initial_option_price) -> np.ndarray
```

If you need to extend an interface, add a new function. Do not modify existing signatures.

---

## Guidance for Future Claude Sessions Before Editing

1. Read this file (CLAUDE.md) fully.
2. Read `writeup/architecture.md` for module-boundary decisions.
3. Check `writeup/sprint_plan.md` for current build status.
4. Run `source .venv/bin/activate && python -m pytest tests/ -v` to confirm tests still pass.
5. Read the specific file you are about to edit and understand its current contracts.
6. Do not add new dependencies without checking `requirements.txt` and `pyproject.toml`.
7. Do not silently expand scope. If a task requires more than was asked, say so.
8. Do not write experiment logic inside `src/`. Do not write library logic inside `experiments/`.
9. When in doubt about a module boundary, check `writeup/architecture.md` first.

---

## Recommended Work Order for the Rest of the Sprint

See `writeup/sprint_plan.md` for the full plan. Short version:

1. Install torch in venv: `source .venv/bin/activate && pip install torch`
2. Build `src/config.py` + `src/risk_measures.py` + `tests/test_risk_measures.py`
3. Build `src/hedger.py` + `tests/test_hedger.py` + `experiments/02_classical_delta.py`
4. Build `experiments/01_validate_market.py` (GBM validation plots)
5. Build `src/neural_hedger.py` + `src/train.py`
6. Build `experiments/03_neural_no_costs.py` — GATE: must recover BS delta before proceeding
7. Build `experiments/04_neural_with_costs.py`
8. Build `experiments/05_cost_frontier.py` — the headline figure
9. Write `writeup/math_reference.md` (user's own derivations)
10. Write `writeup/paper_notes.md` (user's Buehler 2019 annotations)
11. Polish README.md, run `make reproduce`, tag release

---

## What Success Looks Like at the End of the Sprint

- `make test` passes with zero failures
- `make reproduce` regenerates all 5 experiment plots from scratch, end-to-end, without error
- `plots/05_cost_frontier.png` exists and clearly shows the neural hedger Pareto-dominating
  BS delta-hedging as transaction costs rise
- `plots/03_learned_vs_bs_delta.png` shows the neural hedger's hedge ratio ≈ BS delta
  in the zero-cost case (side-by-side heatmap or scatter)
- `writeup/math_reference.md` contains the user's BS PDE derivation in their own words
- `writeup/paper_notes.md` contains substantive notes on Buehler 2019
- README.md renders cleanly on GitHub with the headline figure embedded
- All `src/` modules have type annotations and pass `make typecheck`
- Any quant-firm interviewer reading the README in 15 minutes can understand what was built,
  why it works, and what the honest limitations are
