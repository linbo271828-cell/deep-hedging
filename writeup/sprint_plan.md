# Sprint Execution Plan — Deep Hedging

This document is the ground truth for what has been built and what remains.
Update the status column as tasks complete. Check this before starting any session.

Architecture decisions are locked in `writeup/architecture.md`.
Module responsibilities are locked in `CLAUDE.md`.

---

## Current Status

**Days 1–2 complete.** Foundation is solid:
- `src/market.py` — GBM simulator (exact solution), fully tested
- `src/black_scholes.py` — BS pricer + all Greeks, fully tested
- 14 tests, all passing
- Venv at `.venv/`, Python 3.14.3, NumPy + SciPy + pytest installed
- PyTorch NOT yet installed in venv
- `experiments/`, `writeup/`, `plots/` — all empty

**Architecture and documentation foundation complete (this session):**
- `CLAUDE.md` — written, locked
- `writeup/architecture.md` — written, locked
- `writeup/sprint_plan.md` — this file
- `writeup/math_reference.md` — scaffolded (user fills in)
- `writeup/paper_notes.md` — scaffolded (user fills in)
- `README.md` — scaffolded

---

## Dependency Graph

The build order below respects these dependencies:

```
config.py
    ↓
risk_measures.py ←──── train.py ←──── neural_hedger.py
    ↓                      ↓
hedger.py ──────────── experiments/
    ↓
market.py (done)
black_scholes.py (done)
```

Sequential constraints:
- `hedger.py` requires `black_scholes.py` (done) and `config.py`
- `risk_measures.py` requires nothing (pure functions)
- `train.py` requires `neural_hedger.py`, `risk_measures.py`, `config.py`, `hedger.py`
- `experiments/02` requires `hedger.py`
- `experiments/03` requires `train.py` and is a gate for 04 and 05
- `experiments/04` requires passing gate from 03
- `experiments/05` requires 04

Parallel opportunities (can be done simultaneously by different agents):
- `src/config.py` + `src/risk_measures.py` + `tests/test_risk_measures.py`
- `src/hedger.py` + `tests/test_hedger.py`
- `src/neural_hedger.py` (no dependencies on hedger or risk_measures)
- `writeup/math_reference.md` + `writeup/paper_notes.md` (user, independent)

---

## Task List

### Phase 1 — Install and Config (< 1 hour)

| # | Task | Status | Notes |
|---|---|---|---|
| 1.1 | Install torch in venv | TODO | `pip install torch` in `.venv` |
| 1.2 | Verify torch import works | TODO | `python -c "import torch; print(torch.__version__)"` |
| 1.3 | Write `src/config.py` | TODO | Frozen dataclasses: MarketConfig, TrainingConfig |

---

### Phase 2 — Risk Measures (parallel with Phase 3)

| # | Task | Status | Notes |
|---|---|---|---|
| 2.1 | Write `src/risk_measures.py` | TODO | cvar(), entropic_risk(). Pure numpy functions. |
| 2.2 | Write `tests/test_risk_measures.py` | TODO | CVaR on Gaussian, uniform. Entropic on known dist. |
| 2.3 | Verify tests pass | TODO | `pytest tests/test_risk_measures.py -v` |

Acceptance criteria for CVaR:
- On a standard Gaussian: CVaR_0.95 ≈ 2.063 (known analytical value)
- CVaR is monotonically decreasing in alpha (higher alpha = better tail)
- CVaR of a constant is that constant

---

### Phase 3 — Classical Hedger (parallel with Phase 2)

| # | Task | Status | Notes |
|---|---|---|---|
| 3.1 | Write `src/hedger.py` | TODO | hedge_pnl(), classical_delta_pnl() |
| 3.2 | Write `tests/test_hedger.py` | TODO | See acceptance criteria below |
| 3.3 | Verify tests pass | TODO | `pytest tests/test_hedger.py -v` |
| 3.4 | Write `experiments/01_validate_market.py` | TODO | GBM moment validation plot |
| 3.5 | Write `experiments/02_classical_delta.py` | TODO | Classical hedger P&L distribution |
| 3.6 | Run experiment 02 and verify output | TODO | `python experiments/02_classical_delta.py` |

Acceptance criteria for hedger tests:
- With cost_rate=0.0 and many rebalancing steps: mean(pnl) ≈ 0 (within 2 SE)
- With cost_rate=0.0005 and n_steps=50: mean(pnl) < 0 (costs reduce mean P&L)
- P&L with costs has strictly lower mean than without costs

Acceptance criteria for experiment 02:
- Two P&L histograms: no-cost (centered near 0) and with-cost (shifted negative)
- Printed summary: mean, std, CVaR for both
- Saved to plots/02_pnl_distribution.png

---

### Phase 4 — Neural Hedger Model

| # | Task | Status | Notes |
|---|---|---|---|
| 4.1 | Write `src/neural_hedger.py` | TODO | HedgeNet(n_layers, hidden_dim). PyTorch. |
| 4.2 | Write `src/train.py` | TODO | train(config, market_config, model, optimizer) |
| 4.3 | Manual smoke test | TODO | Instantiate model, run one forward pass, check output shape |

No formal unit tests for train.py (training loops are hard to unit-test).
The acceptance gate is experiment 03 (below).

---

### Phase 5 — Zero-Cost Gate (BLOCKING)

**This is the most important gate in the project. Do not proceed to Phase 6 without passing it.**

| # | Task | Status | Notes |
|---|---|---|---|
| 5.1 | Write `experiments/03_neural_no_costs.py` | TODO | Train with cost_rate=0, compare to BS delta |
| 5.2 | Run experiment 03 | TODO | Should take 5–15 min on CPU |
| 5.3 | Verify gate: learned delta ≈ BS delta | TODO | GATE — see criteria below |

Gate criteria:
- Plot `plots/03_learned_vs_bs_delta.png`: side-by-side heatmap or scatter of
  HedgeNet output vs. BS call_delta over a grid of (S, tau) values
- Mean absolute deviation between learned and BS delta < 0.05 across the grid
- This must hold across the (S, tau) grid, not just at ATM

What to do if the gate fails:
1. Check learning rate (try 1e-3 → 5e-4)
2. Check n_epochs (try 200 → 500)
3. Check n_paths (try 50k → 100k)
4. Check feature normalization (S_t/S_0 must be near 1, not raw 100)
5. Check the loss is actually differentiable through the P&L (no detached tensors)
6. Debug by plotting the training loss curve first — if it's not decreasing, the
   problem is in the training loop, not the architecture

**Do NOT proceed to Phase 6 if this gate fails.**

---

### Phase 6 — Transaction Cost Experiments

| # | Task | Status | Notes |
|---|---|---|---|
| 6.1 | Write `experiments/04_neural_with_costs.py` | TODO | Retrain with cost_rate=0.0005 (5 bps) |
| 6.2 | Run experiment 04 | TODO | Verify deviation from BS delta |
| 6.3 | Write `experiments/05_cost_frontier.py` | TODO | Sweep cost_rate, plot CVaR vs. expected cost |
| 6.4 | Run experiment 05 | TODO | The headline figure |

Acceptance criteria for experiment 04:
- The learned hedge ratio with costs is smoother (less reactive) than BS delta
- The deviation from BS delta is largest near ATM where delta changes most
- Printed summary: CVaR and expected cost for neural hedger vs. classical hedger at 5 bps

Acceptance criteria for experiment 05:
- X-axis: expected transaction cost per path
- Y-axis: CVaR of terminal P&L at 95%
- Two curves: classical delta hedger, neural hedger
- Neural hedger curve should be Pareto-dominant (lower CVaR at same cost) as costs rise
- The two curves should coincide at cost_rate=0 (both approach the same hedge)
- Saved to plots/05_cost_frontier.png — this is the headline figure for the README

---

### Phase 7 — Polish and Writeup

| # | Task | Status | Notes |
|---|---|---|---|
| 7.1 | Fill in `writeup/math_reference.md` | TODO | User writes their own derivations |
| 7.2 | Fill in `writeup/paper_notes.md` | TODO | User reads Buehler 2019, takes notes |
| 7.3 | Add headline figure to README | TODO | `![cost frontier](plots/05_cost_frontier.png)` |
| 7.4 | Run `make lint` and fix all warnings | TODO | `ruff check src tests experiments` |
| 7.5 | Run `make typecheck` and fix all errors | TODO | `mypy src` |
| 7.6 | Run `make reproduce` end-to-end | TODO | Must complete without errors |
| 7.7 | Update CLAUDE.md status table | TODO | Mark all completed files |
| 7.8 | Tag git release v0.1.0 | TODO | `git tag v0.1.0` |

---

### Stretch Goals (Only if Days 1–10 finish ahead of schedule)

| # | Task | Priority |
|---|---|---|
| S.1 | Heston stochastic volatility market model | Medium |
| S.2 | GitHub Actions CI — run tests on push | Low |
| S.3 | 4–6 page report PDF | Medium |

**Do not start stretch goals before experiment 05 is complete and passing.**

---

## Key Numerical Expectations (Reference)

These are the claims the project makes. If any of these fail, there is a bug.

| Claim | Expected value | Where verified |
|---|---|---|
| GBM log-return mean | (mu - sigma^2/2)*T | test_market.py, exp 01 |
| GBM log-return variance | sigma^2 * T | test_market.py, exp 01 |
| BS delta at ATM (S=K, tau=0.5, r=0.02, sigma=0.2) | ≈ 0.534 | test_black_scholes.py |
| MC price (200k paths) matches BS | within 3 SE | test_black_scholes.py |
| Classical hedger P&L mean (no cost) | ≈ 0 | test_hedger.py |
| Neural hedger vs BS delta (no cost) | MAD < 0.05 | exp 03 gate |
| CVaR (Gaussian, 95%) | ≈ 2.063 | test_risk_measures.py |

---

## How to Run Everything

```bash
# One-time setup
source .venv/bin/activate
pip install torch  # if not yet installed

# Tests (should always be green)
pytest tests/ -v

# Individual experiments
python experiments/01_validate_market.py
python experiments/02_classical_delta.py
python experiments/03_neural_no_costs.py
python experiments/04_neural_with_costs.py
python experiments/05_cost_frontier.py

# Full reproduce (tests + all experiments)
make reproduce
```

All plots are saved to `plots/`. All numerical summaries print to stdout.
