# CLAUDE.md — Deep Hedging Lab

This file is the primary operating guide for any Claude Code session working on this repo.
Read it fully before touching any file.

---

## Project Mission

Deep Hedging Lab is a friction-aware hedging research suite that compares classical
and learned hedging policies across market models, payoff families, hedge universes,
risk objectives, and transaction-cost regimes.

Inspired by: Buehler, Gonon, Teichmann, Wood — *Deep Hedging* (Quantitative Finance 2019).

This is a **quant-finance portfolio project** targeting quant research / trading internships
(Jane Street, Citadel, Two Sigma, Optiver, Hudson River, DRW).
It demonstrates: mathematical modeling, faithful paper implementation, disciplined engineering,
and honest statistical evaluation.

---

## What This Project Is and Is Not

**Is:**
- A multi-model, multi-payoff, multi-universe comparison of classical vs. learned hedging
- A reproducible research system with named experiments and saved artifacts
- An honest implementation with clearly stated limitations

**Is not:**
- A live trading system
- A web application
- A production-grade calibration framework
- A proof that neural hedging always beats classical hedging

---

## v1 Scope

**In scope:**
- Market models: GBM, Heston
- Payoffs: European call, put, vertical spread, straddle
- Hedge universes: stock-only, stock + one liquid option
- Risk objectives: CVaR, entropic risk
- Frictions: proportional transaction costs only
- Neural hedger families: feedforward networks (small, medium, position-aware, Heston-aware)
- Reproducible experiment registry, named configs, saved artifacts
- Summary tables and markdown report

**Explicitly out of scope for v1:**
- American/Bermudan options
- Local volatility, stochastic interest rates, market impact
- Recurrent/transformer architectures
- Real market data or calibration
- Continuous-time limit analysis
- Web dashboard, Docker, distributed training

Do NOT add features beyond v1 scope without explicit user instruction.

---

## Repository Status

| Module | Status | Notes |
|---|---|---|
| src/models/gbm.py | COMPLETE | Migrated from src/market.py |
| src/models/heston.py | STUB | Phase C |
| src/payoffs/european.py | COMPLETE | Migrated from src/black_scholes.py |
| src/payoffs/spreads.py | STUB | Phase B |
| src/payoffs/straddles.py | STUB | Phase B |
| src/hedging/pnl.py | COMPLETE | Migrated from src/hedger.py |
| src/hedging/transaction_costs.py | COMPLETE | Extracted from hedger |
| src/hedging/baseline.py | COMPLETE | Migrated from src/hedger.py |
| src/hedging/hedge_universe.py | COMPLETE | StockOnlyUniverse, StockPlusOptionUniverse, make_universe |
| src/risk/cvar.py | COMPLETE | Migrated from src/risk_measures.py |
| src/risk/entropic.py | COMPLETE | Migrated from src/risk_measures.py |
| src/neural/architectures.py | COMPLETE | Migrated from src/neural_hedger.py |
| src/neural/features.py | PARTIAL | base_features complete; heston_features stub |
| src/neural/train.py | COMPLETE | Migrated from src/train.py |
| src/neural/evaluate.py | COMPLETE | Phase 1 — evaluate() implemented |
| src/experiments/configs.py | COMPLETE | Expanded from src/config.py |
| src/experiments/registry.py | PARTIAL | Phase A baseline configs registered |
| src/experiments/runner.py | COMPLETE | Phase 1 — run_experiment() implemented |
| src/experiments/summaries.py | COMPLETE | SummaryRow schema defined |
| src/reporting/tables.py | STUB | Phase E |
| src/reporting/plots.py | STUB | Phase E |
| src/reporting/report.py | STUB | Phase E |
| src/utils/seeds.py | COMPLETE | derive_seed utility |
| src/utils/io.py | PARTIAL | save/load JSON complete |
| src/utils/stats.py | COMPLETE | standard_error, report_moment |
| src/market.py | SHIM | → src/models/gbm.py |
| src/black_scholes.py | SHIM | → src/payoffs/european.py |
| src/hedger.py | SHIM | → src/hedging/pnl.py + baseline.py |
| src/risk_measures.py | SHIM | → src/risk/ |
| src/config.py | SHIM | → src/experiments/configs.py |
| src/neural_hedger.py | SHIM | → src/neural/architectures.py |
| src/train.py | SHIM | → src/neural/train.py |
| experiments/01-05 | COMPLETE | v0.1 baseline, still running |
| tests/test_neural_smoke.py | COMPLETE | 7 neural smoke tests |
| tests/test_runner_smoke.py | COMPLETE | 13 runner smoke tests |
| tests/test_api_smoke.py | COMPLETE | 17 FastAPI smoke tests (incl. surface) |
| tests/test_hedge_universe.py | COMPLETE | 17 stock+option smoke tests |
| tests/ | COMPLETE | 134 tests, all passing |
| services/sim/main.py | COMPLETE | FastAPI backend — 7 routes incl. /surface; stock+option presets |
| apps/web/ | COMPLETE | Next.js frontend — dark theme, landing + results + 3D surface |
| apps/web/components/HedgeSurface.tsx | COMPLETE | Interactive 3-mode Plotly 3D surface |
| results/raw/ | EMPTY | populated by runner |
| results/processed/ | EMPTY | populated by runner |
| results/reports/ | EMPTY | populated by reporting |
| writeup/architecture.md | UPDATED | Deep Hedging Lab architecture |
| writeup/experiment-contracts.md | NEW | Config and artifact schema |
| writeup/ownership.md | NEW | 7-lane ownership boundaries |
| writeup/implementation_plan.md | NEW | Phase-by-phase plan |

Update this table whenever a file changes status.

---

## Migration Notes from deep-hedging v0.1

The v0.1 flat `src/*.py` modules are now **compatibility shims**.
They re-export from the canonical new locations in subdirectories.
Do NOT add new logic to the shims.
The shims will be removed once all experiments are migrated to use the new paths.

Canonical new import paths:
```python
from src.models.gbm import GBM, time_grid
from src.payoffs.european import call_price, call_delta
from src.hedging.pnl import hedge_pnl
from src.hedging.baseline import classical_delta_pnl
from src.risk.cvar import cvar
from src.risk.entropic import entropic_risk
from src.neural.architectures import HedgeNet
from src.neural.train import train
from src.experiments.configs import MarketConfig, TrainingConfig, ExperimentConfig
```

Old import paths (shims, backward-compatible):
```python
from src.market import GBM, time_grid         # → src/models/gbm.py
from src import black_scholes as bs            # → src/payoffs/european.py
from src.hedger import hedge_pnl               # → src/hedging/pnl.py
from src.risk_measures import cvar             # → src/risk/cvar.py
from src.config import MarketConfig            # → src/experiments/configs.py
```

---

## Directory Map

```
deep-hedging/
├── CLAUDE.md                       # This file. Read first.
├── README.md                       # Public-facing project writeup
├── spec.md                         # Full product and engineering spec
├── pyproject.toml                  # Project config + tool settings
├── requirements.txt                # Flat dep list for venv install
├── Makefile                        # make test / lint / reproduce-core / reproduce-report
├── .gitignore
│
├── src/                            # All importable library code
│   ├── __init__.py
│   ├── models/                     # Market simulators
│   │   ├── gbm.py                  # GBM exact-solution sampler (COMPLETE)
│   │   └── heston.py               # Heston SV model (STUB)
│   ├── payoffs/                    # Terminal payoffs + closed-form pricing
│   │   ├── european.py             # BS pricing + European call/put payoffs (COMPLETE)
│   │   ├── spreads.py              # Vertical spreads (STUB)
│   │   └── straddles.py            # Straddles (STUB)
│   ├── hedging/                    # P&L engine, costs, baselines
│   │   ├── pnl.py                  # Canonical P&L engine (COMPLETE)
│   │   ├── transaction_costs.py    # Cost models (COMPLETE)
│   │   ├── baseline.py             # Classical BS delta hedger (COMPLETE)
│   │   └── hedge_universe.py       # Hedge universe protocol + stubs
│   ├── risk/                       # Risk measures
│   │   ├── cvar.py                 # CVaR / Expected Shortfall (COMPLETE)
│   │   └── entropic.py             # Entropic risk (COMPLETE)
│   ├── neural/                     # Neural hedger stack
│   │   ├── architectures.py        # HedgeNet FFN (COMPLETE)
│   │   ├── features.py             # Feature builder utilities (PARTIAL)
│   │   ├── train.py                # CVaR training loop (COMPLETE)
│   │   └── evaluate.py             # Evaluation (STUB)
│   ├── experiments/                # Config system
│   │   ├── configs.py              # Frozen dataclasses (COMPLETE)
│   │   ├── registry.py             # Named experiment registry (PARTIAL)
│   │   ├── runner.py               # Experiment runner (STUB)
│   │   └── summaries.py            # SummaryRow schema (COMPLETE)
│   ├── reporting/                  # Report generation
│   │   ├── tables.py               # Summary tables (STUB)
│   │   ├── plots.py                # Plot generation (STUB)
│   │   └── report.py               # Markdown report (STUB)
│   └── utils/                      # Shared utilities
│       ├── seeds.py                # Seed derivation (COMPLETE)
│       ├── io.py                   # Artifact I/O (PARTIAL)
│       └── stats.py                # Standard error etc. (COMPLETE)
│
│   [SHIMS — backward compatibility only, do not add logic here]
│   ├── market.py                   # → src/models/gbm.py
│   ├── black_scholes.py            # → src/payoffs/european.py
│   ├── hedger.py                   # → src/hedging/
│   ├── risk_measures.py            # → src/risk/
│   ├── config.py                   # → src/experiments/configs.py
│   ├── neural_hedger.py            # → src/neural/architectures.py
│   └── train.py                    # → src/neural/train.py
│
├── experiments/                    # Named experiment entrypoints
│   ├── 01_validate_market.py       # GBM moment validation (v0.1 baseline)
│   ├── 02_classical_delta.py       # Classical delta hedger P&L
│   ├── 03_neural_no_costs.py       # Neural hedger zero-cost gate
│   ├── 04_neural_with_costs.py     # Neural hedger under costs
│   └── 05_cost_frontier.py         # Cost-frontier sweep (headline figure)
│
├── tests/                          # pytest — numerical correctness
│   ├── test_market.py              # 7 GBM tests
│   ├── test_black_scholes.py       # 7 BS tests
│   ├── test_hedger.py              # 5 P&L tests
│   └── test_risk_measures.py       # 15 risk-measure tests
│
├── results/                        # Generated artifacts (gitignored except .gitkeep)
│   ├── raw/                        # One JSON per run (config + metrics)
│   ├── processed/                  # Aggregated summary CSV/JSON
│   └── reports/                    # Generated markdown report + figures
│
├── plots/                          # Legacy plot output for v0.1 experiments
│
├── notebooks/                      # Scratch derivations only
│   └── 00_derivations.ipynb
│
└── writeup/                        # Architecture + math documentation
    ├── architecture.md             # Module boundaries (this session, UPDATED)
    ├── experiment-contracts.md     # Config + artifact schema (NEW)
    ├── ownership.md                # 7-lane ownership boundaries (NEW)
    ├── implementation_plan.md      # Phase-by-phase build plan (NEW)
    ├── math_reference.md           # Author's BS derivations
    ├── paper_notes.md              # Buehler 2019 annotations
    └── sprint_plan.md              # v0.1 sprint plan (historical)
```

---

## Core Pipeline

```
market model → hedge universe → payoff → hedge policy → P&L engine → risk measure
     ↓                                                                      ↓
src/models/                                                         src/risk/
src/hedging/hedge_universe.py                                experiments/
src/payoffs/                                                 results/
src/neural/ or src/hedging/baseline.py
src/hedging/pnl.py
```

---

## Shared Contract Rules (Do Not Break Without Coordinating)

The following function signatures are locked. Do not change them without
updating ALL callers and notifying the lead:

```python
# src/models/gbm.py
GBM.sample_paths(n_paths, n_steps, T, seed) -> np.ndarray  # (n_paths, n_steps+1)

# src/payoffs/european.py
call_price(s, k, r, sigma, tau) -> np.ndarray | float
call_delta(s, k, r, sigma, tau) -> np.ndarray | float

# src/hedging/pnl.py
hedge_pnl(paths, delta_fn, time_grid, k, r, sigma, cost_rate, initial_option_price) -> np.ndarray

# src/risk/cvar.py
cvar(pnl, alpha=0.95) -> float

# src/risk/entropic.py
entropic_risk(pnl, lambda_=1.0) -> float

# src/experiments/configs.py
MarketConfig, TrainingConfig  — no field removal or rename without updating all callers
```

If you need to extend an interface, add a new function. Do not modify existing signatures.

---

## Ownership Lanes (summary)

See `writeup/ownership.md` for full details.

| Lane | Who | Primary files |
|---|---|---|
| Lead | Architect | Architecture, shared contracts, registry, integration |
| A | Market models | src/models/, tests/test_market.py |
| B | Hedging core | src/hedging/, tests/test_hedger.py |
| C | Payoffs + baselines | src/payoffs/, src/hedging/baseline.py |
| D | Neural | src/neural/, smoke tests |
| E | Experiment system | src/experiments/, experiments/ |
| F | Reporting/docs | src/reporting/, writeup/, README.md |
| G | QA/repro | tests/, Makefile, lint/typecheck |

---

## Coding Standards

- Python 3.11 type annotations on all public functions. Use `np.ndarray` not `NDArray`.
- Line length: 100 (configured in pyproject.toml).
- ruff rules: E, F, I, N, UP, B, SIM, RUF. Run `make lint` before committing.
- mypy strict mode. All `src/` must type-check clean. Tests excluded from strict mypy.
- Docstrings on all public functions: one-line summary, key math in plain text.
- No `print()` in `src/`. Use stdout only in `experiments/`.
- No global mutable state. Pass seeds explicitly everywhere.
- No `import *`.
- Do not add dependencies without approval.

---

## Numerical Rigor Rules

- GBM paths: exact exponential solution. No Euler-Maruyama. Non-negotiable.
- Black-Scholes formulas must match finite-difference Greeks to test tolerances.
- Monte Carlo results must agree with closed-form within 3 SE at 200k paths.
- Always compute and report standard errors alongside empirical moments.
- Never use `np.random.seed()`. Always use `np.random.default_rng(seed)`.
- Never use `torch.manual_seed()` globally. Call it at the top of `train()` only.
- All float comparisons in tests: `np.testing.assert_allclose` with explicit `atol`/`rtol`.

---

## Seed Discipline Rules

- NumPy: `rng = np.random.default_rng(seed)`. Never re-use `rng` across functions.
- Derived seeds: `from src.utils.seeds import derive_seed` → `derive_seed(base, "train")`.
- PyTorch: `torch.manual_seed(config.seed)` called once at top of `train()`.
- Training and validation paths must use different seeds.
- Never share seeds between experiments meant to be independent.

---

## Reproducibility Rules

- Every experiment has a named config in the registry.
- Configs record all seeds, hyperparameters, market parameters.
- `make reproduce-core` regenerates the v0.1 GBM-call baseline from scratch.
- `make reproduce-report` regenerates the markdown report from saved results.
- Raw results are saved to `results/raw/<name>/` as JSON.
- Processed summaries are saved to `results/processed/`.
- Plots are saved to `results/reports/` (via reporting pipeline) or `plots/` (legacy v0.1).

---

## Experiment Conventions

- Scripts in `experiments/` are named `{NN}_{descriptor}.py`.
- Each script: instantiates a config, does work, saves output, prints summary.
- No argparse. Configs live in `src/experiments/configs.py`.
- New experiments should use `src/experiments/registry.py` to resolve configs.
- Outputs: artifacts → `results/raw/`, plots → `results/reports/`, summary → stdout.
- v0.1 legacy scripts (01-05) still output to `plots/` for backward compatibility.

---

## Testing Philosophy

- Tests verify mathematical claims, not implementation details.
- No mocking of numerical functions. Test against closed-form expectations.
- Total test suite under 60 seconds.
- Neural hedger recovering BS delta in zero-cost case is an experiment gate, not a unit test.
- Smoke tests (training loop, forward pass, runner) belong in tests/ as well.

---

## Before Editing Any File

1. Read this file fully.
2. Read `writeup/architecture.md` for the module boundary decisions.
3. Run `source .venv/bin/activate && python -m pytest tests/ -v` to confirm tests still pass.
4. Read the specific file you are about to modify to understand its current contracts.
5. Check `writeup/ownership.md` to confirm you are editing files in your lane.
6. If editing a shared-contract file, check ALL callers before changing any signature.
7. Do not silently expand scope. If a task requires more than was asked, say so.
8. Do not write experiment logic inside `src/`. Do not write library logic in `experiments/`.

---

## Recommended Next Work Order

See `writeup/implementation_plan.md` for the detailed plan. Short version:

**Phase 1 — Baseline migration (now)**
- Implement `src/neural/evaluate.py`
- Implement `src/experiments/runner.py`
- New experiment scripts that use the registry + runner pattern
- Migrate experiments/01-05 to use new import paths (optional, shims work)

**Phase 2 — Multi-payoff**
- `src/payoffs/spreads.py` — call_spread_payoff, put_spread_payoff
- `src/payoffs/straddles.py` — straddle_payoff
- New experiments 10-14 for put, spread, straddle under GBM

**Phase 3 — Heston**
- `src/models/heston.py` — Andersen QE or Euler simulation
- `src/neural/features.py` — heston_features
- New experiments for Heston model variants

**Phase 4 — Stock+option hedge universe**
- `src/hedging/hedge_universe.py` — StockOnlyUniverse, StockPlusOptionUniverse
- Multi-instrument P&L engine extension

**Phase 5 — Ablations + reporting**
- Architecture size, feature ablation, objective comparison experiments
- `src/reporting/` — tables, plots, report generation
- README update with full results table

---

## What Good Looks Like at v1 Completion

- `make test` passes zero failures
- `make reproduce-core` regenerates all 5 v0.1 plots end-to-end
- `make reproduce-report` generates the full markdown report from saved results
- All 15+ experiments produce artifacts in `results/raw/`
- `results/reports/report.md` contains the complete comparison table
- All `src/` passes `make typecheck` with zero errors
- Any quant interviewer reading the README in 15 minutes understands the scope,
  results, and honest limitations
