# Deep Hedging Lab — Product & Engineering Spec

## 1. Overview

### 1.1 Working title
Deep Hedging Lab

### 1.2 One-line pitch
A friction-aware hedging research suite that compares classical and learned hedging policies across market models, payoff families, hedge universes, risk objectives, and transaction-cost regimes.

### 1.3 Project thesis
The existing `deep-hedging` repo proves a narrow result:
- one market model
- one primary payoff
- one hedge universe
- one main baseline comparison

Deep Hedging Lab generalizes that into a compact research system that can answer:
- when learned hedging matches classical hedging
- when it diverges
- when that divergence is beneficial
- how robust it is to model choice, costs, payoff shape, and state design

This project is inspired by the deep hedging framework of Buehler, Gonon, Teichmann, and Wood: hedging under frictions as direct optimization of trading strategies under convex risk measures rather than relying only on frictionless replication.

### 1.4 Core pipeline
market model -> tradable hedge universe -> payoff -> hedge policy -> P&L engine -> risk evaluation -> experiment summaries -> plots/report

---

## 2. Goals

### 2.1 Primary goals
- Build a significantly more ambitious follow-up to `deep-hedging`
- Turn a single-demo repo into a compact hedging research suite
- Compare learned hedging across:
  - GBM vs Heston
  - multiple payoffs
  - stock-only vs stock+option hedge sets
  - CVaR vs entropic-risk objectives
  - cost sweeps and feature ablations
- Make the codebase modular enough for larger Claude agent-team delegation
- Keep the project mathematically grounded, reproducible, and honest

### 2.2 Secondary goals
- Produce clean summary tables and reproducible reports
- Make the repo feel like a research system, not a pile of scripts
- Preserve a strong interview story for quant research / trading / quant dev
- Keep scope bounded enough to finish

### 2.3 Non-goals
Deep Hedging Lab v1 is not trying to:
- be a live trading system
- be a web app
- perform production-grade market calibration
- handle American options, stochastic rates, local vol calibration, or market impact in v1
- prove that neural hedging universally dominates classical hedging
- become a giant framework with unnecessary abstraction

---

## 3. Core research questions

### RQ1 — Model dependence
How does learned hedging differ under GBM vs Heston?

### RQ2 — Friction sensitivity
How robust is neural hedging across proportional transaction-cost regimes?

### RQ3 — Payoff dependence
Do learned hedging gains differ across calls, puts, spreads, and straddles?

### RQ4 — Hedge-universe dependence
When does stock+option hedging materially improve outcomes vs stock-only hedging?

### RQ5 — Architecture and feature dependence
How sensitive are results to network architecture and state inputs?

### RQ6 — Objective dependence
How different are learned policies under CVaR vs entropic risk?

---

## 4. Scope for v1

### 4.1 Market models
- GBM
- Heston

### 4.2 Payoffs
- European call
- European put
- vertical spread
- straddle

### 4.3 Hedge universes
- stock-only
- stock + one liquid option

### 4.4 Risk objectives
- CVaR
- entropic risk

### 4.5 Frictions
- proportional transaction costs only

### 4.6 Neural hedger families
At minimum:
- small feedforward network
- medium feedforward network
- position-aware feedforward network
- Heston-aware feature variant

### 4.7 Outputs
- reproducible experiment runs
- saved config + metrics artifacts
- summary CSV / JSON tables
- plots
- markdown report generated from results

---

## 5. Out of scope for v1

Do not build in v1:
- American or Bermudan exercise
- local volatility
- stochastic interest rates
- market impact
- reinforcement-learning toolkit abstractions
- transformer hedgers
- historical live-data ingestion
- web dashboard
- dozens of architecture variants

---

## 6. Product identity and positioning

Deep Hedging Lab should be presented as:

> A friction-aware hedging research suite for comparing classical and learned hedge policies across market models, instruments, payoff families, and convex risk objectives.

It should feel:
- rigorous
- modular
- reproducible
- compact
- research-first
- honest about limitations

It should not feel:
- overhyped
- overengineered
- like a generic ML benchmark zoo

---

## 7. High-level architecture

### 7.1 Target structure
```text
deep-hedging/
  src/
    models/
    payoffs/
    hedging/
    neural/
    risk/
    experiments/
    reporting/
    utils/
  experiments/
  results/
    raw/
    processed/
    reports/
  tests/
  writeup/
  README.md
  CLAUDE.md
  spec.md
```

### 7.2 Design principle
Library code lives in `src/`.
Named experiment entrypoints live in `experiments/`.
Generated artifacts live in `results/`.
Plots and report tables should be generated from saved run outputs, not hand-maintained.

### 7.3 Shared contracts
The most important shared contracts are:
- market path interface
- payoff interface
- P&L engine inputs/outputs
- risk metric interface
- experiment config interface
- experiment summary row schema

These should be stable and centrally documented.

---

## 8. Module plan

### 8.1 `src/models/`
- `gbm.py`
- `heston.py`

Shared requirement:
Both models should expose a common interface returning a `MarketPaths` object or equivalent typed structure.

### 8.2 `src/payoffs/`
- `european.py`
- `spreads.py`
- `straddles.py`

Requirements:
- pure vectorized functions
- no model-specific code
- no training logic

### 8.3 `src/hedging/`
- `pnl.py` — canonical P&L engine
- `transaction_costs.py`
- `hedge_universe.py`
- `baseline.py`

Rule:
If no exact classical benchmark exists in a setting, the repo must say that explicitly.

### 8.4 `src/neural/`
- `architectures.py`
- `features.py`
- `train.py`
- `evaluate.py`

Rule:
Training and evaluation must be separate.

### 8.5 `src/risk/`
- `cvar.py`
- `entropic.py`

### 8.6 `src/experiments/`
- `configs.py`
- `registry.py`
- `runner.py`
- `summaries.py`

### 8.7 `src/reporting/`
- `tables.py`
- `plots.py`
- `report.py`

### 8.8 `src/utils/`
- `seeds.py`
- `io.py`
- `stats.py`

---

## 9. Metrics

Every experiment should output:

### Core metrics
- mean P&L
- standard deviation of P&L
- CVaR
- entropic risk
- expected transaction cost
- turnover
- average absolute hedge adjustment
- hedge tracking error vs baseline where applicable

### Secondary metrics
- training runtime
- evaluation runtime
- parameter count
- repeated-seed variability
- convergence diagnostics

### Win criterion
A learned hedge is meaningfully better only if it improves the target risk/cost tradeoff in a stable out-of-sample way, ideally across repeated seeds.

---

## 10. Experimental program

### Phase A — Baseline migration
Reproduce current `deep-hedging` results inside the new architecture.

Experiments:
1. GBM + call + stock-only + zero cost
2. GBM + call + stock-only + cost sweep

Gate:
- zero-cost learned hedge approximates BS delta
- current headline cost result is recovered or closely matched

### Phase B — Multi-payoff study
Experiments:
3. GBM + put + stock-only
4. GBM + spread + stock-only
5. GBM + straddle + stock-only

### Phase C — Heston study
Experiments:
6. Heston + call + stock-only + zero cost
7. Heston + call + stock-only + cost sweep
8. Heston + spread + stock-only

### Phase D — Hedge-universe study
Experiments:
9. GBM + call + stock-only vs stock+option
10. Heston + call + stock-only vs stock+option
11. Heston + spread + stock-only vs stock+option

### Phase E — Ablations
Experiments:
12. architecture size comparison
13. feature ablation
14. objective comparison
15. position-aware vs not

---

## 11. Benchmarking philosophy

The repo must never imply:
- neural hedging always beats classical hedging
- model-free means benchmark-free
- learned policies are universally superior

Instead:
- under frictionless simple models, learned hedging should recover the classical solution
- under frictions or richer dynamics, learned policies may outperform classical frictionless heuristics
- results depend on regime, objective, and hedge universe

---

## 12. Reproducibility requirements

Every experiment must:
- have a named config
- record seed(s)
- save metrics and config artifacts
- save plots deterministically
- be rerunnable from a clean checkout

Required commands:
- `make test`
- `make lint`
- `make typecheck`
- `make reproduce-core`
- `make reproduce-report`

Repeated-seed studies should use fixed predefined seed lists.

---

## 13. Testing requirements

### Unit tests
- market simulators
- payoff functions
- transaction costs
- P&L engine
- risk metrics
- feature builders

### Smoke tests
- neural architecture forward pass
- training loop short-run smoke tests
- experiment runner smoke tests

### Regression tests
- current GBM-call no-cost sanity check
- one cost-sweep experiment summary shape
- report generation smoke test

---

## 14. Build phases

### Phase 0
Create `deep-hedging-lab` direction from current `deep-hedging` baseline.

### Phase 1
Architecture migration:
- modular structure
- shared interfaces
- current results reproduced

### Phase 2
Multi-payoff support.

### Phase 3
Heston support.

### Phase 4
Stock+option hedge universe.

### Phase 5
Ablations, summaries, reporting.

---

## 15. Claude agent-team development plan

This project is broad enough for 6–7 implementation lanes.

### Lead
Owns:
- architecture
- shared contracts
- integration
- experiment registry
- final report structure

### Lane A — market models
Owns:
- `src/models/*`
- model tests

### Lane B — hedging core
Owns:
- `src/hedging/*`
- transaction costs
- P&L engine
- tests

### Lane C — payoffs and baselines
Owns:
- `src/payoffs/*`
- `src/hedging/baseline.py`
- payoff tests

### Lane D — neural
Owns:
- `src/neural/*`
- train/evaluate
- smoke tests

### Lane E — experiment system
Owns:
- `src/experiments/*`
- experiment configs
- runner
- summaries

### Lane F — reporting/docs
Owns:
- `src/reporting/*`
- README
- writeup
- report generation

### Lane G — QA/repro
Owns:
- full test suite
- lint/typecheck
- Makefile / CI
- reproduce commands

---

## 16. Final implementation choices to lock now

For Deep Hedging Lab v1, lock:
- Python-only research repo
- GBM + Heston
- call + put + spread + straddle
- stock-only + stock+one-option
- CVaR + entropic
- proportional transaction costs only
- reproducible experiment registry
- summary tables and markdown report
- no web app
- no market impact
- no recurrent nets in v1

---

## 17. Final positioning statement

Deep Hedging Lab should be presented as:

> A multi-model, friction-aware hedging research suite for studying when and why learned hedge policies differ from classical hedging across market dynamics, payoff shapes, instrument sets, and convex risk objectives.
