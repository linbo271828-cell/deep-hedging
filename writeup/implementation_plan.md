# Implementation Plan — Deep Hedging Lab

This document specifies the phase-by-phase build plan for Deep Hedging Lab v1.
It corresponds to the experimental program in spec.md §10 and §14.

Each phase has entry criteria, exit criteria, dependencies, parallelization notes,
and what must remain sequential.

---

## Phase 0 — Architecture Migration (COMPLETE)

**What**: Migrate from flat deep-hedging v0.1 to the Deep Hedging Lab scaffold.

**Entry criteria**: v0.1 is complete with 34 passing tests.

**Exit criteria** (all met):
- New directory structure created (src/models/, src/payoffs/, etc.)
- All v0.1 code moved to canonical locations
- Flat src/*.py converted to compatibility shims
- 34 tests still passing after migration
- CLAUDE.md, architecture.md, ownership.md, experiment-contracts.md written
- This implementation_plan.md written

**Status**: COMPLETE.

---

## Phase 1 — Baseline Migration

**Goal**: Reproduce v0.1 GBM-call results inside the new architecture.
Correspond to spec.md Phase A experiments (exp 1 and 2).

**Experiments to produce**:
1. GBM + call + stock-only + zero cost (neural recovers BS delta)
2. GBM + call + stock-only + cost sweep (headline cost-frontier figure)

**Entry criteria**:
- Phase 0 complete
- 34 tests passing

**Exit criteria**:
- `src/neural/evaluate.py` implemented and tested
- `src/experiments/runner.py` implemented and tested
- Experiments 1 and 2 (above) run via `runner.run_experiment("gbm_call_no_cost")` etc.
- SummaryRows saved to `results/raw/`
- Zero-cost gate: neural MAD vs BS delta < 0.05 (same as v0.1 gate)
- Cost-frontier result closely matches v0.1 `plots/05_cost_frontier.png`
- `make reproduce-core` still passes (v0.1 scripts still work via shims)

**Lanes**:
- Lane D: implement evaluate.py
- Lane E: implement runner.py; write new experiment scripts using runner
- Lane G: add smoke tests for runner and evaluate

**What can be parallelized**:
- evaluate.py (Lane D) and runner.py (Lane E) are independent
- Smoke tests (Lane G) can be written while implementations are in progress

**What must be sequential**:
- runner.py depends on evaluate.py being importable (even stub-complete)
- End-to-end test of runner depends on both evaluate.py and runner.py

---

## Phase 2 — Multi-Payoff Study

**Goal**: Extend to European put, vertical spread, and straddle under GBM.
Corresponds to spec.md Phase B experiments (exp 3–5).

**Experiments to produce**:
3. GBM + put + stock-only
4. GBM + call_spread + stock-only
5. GBM + straddle + stock-only

**Entry criteria**:
- Phase 1 complete
- Runner produces SummaryRows for Phase 1 experiments

**Exit criteria**:
- `src/payoffs/spreads.py` implemented: call_spread_payoff, put_spread_payoff
- `src/payoffs/straddles.py` implemented: straddle_payoff
- PayoffConfig.payoff_type is used in the runner/P&L engine
- New experiment scripts for exp 3–5
- For each payoff: neural hedger trained, classical baseline computed
- Classical baseline clearly states when it is an approximation (spreads/straddles)
- New payoff tests in tests/test_payoffs.py

**Lanes**:
- Lane C: implement spreads.py, straddles.py, tests
- Lane B: extend pnl.py to accept a general payoff_fn (or modify accounting for new payoffs)
- Lane E: add new experiment configs to registry, new scripts
- Lane G: add payoff tests

**Critical design decision for Lane B**:
`hedge_pnl` currently hardcodes `payoff = max(S_T - K, 0)` at line 112.
For multi-payoff support it needs to accept a `payoff_fn: Callable` parameter.
This is a contract change — coordinate with Lead before implementing.

**What can be parallelized**:
- spreads.py and straddles.py (independent)
- payoff tests can be written alongside implementations

**What must be sequential**:
- pnl.py payoff_fn extension must happen before Lane E can write new experiment scripts
- Lead must approve the pnl.py signature change before Lane B implements it

---

## Phase 3 — Heston Study

**Goal**: Add Heston stochastic volatility market model.
Corresponds to spec.md Phase C experiments (exp 6–8).

**Experiments to produce**:
6. Heston + call + stock-only + zero cost
7. Heston + call + stock-only + cost sweep
8. Heston + spread + stock-only

**Entry criteria**:
- Phase 1 complete (baseline migration done)
- Phase 2 optionally in progress (can be parallelized with Phase 3)

**Exit criteria**:
- `src/models/heston.py` implemented and tested:
  - sample_paths matches theoretical moments
  - sample_paths_with_variance returns both paths
  - Feller condition documented and checked
- `src/neural/features.py` heston_features implemented
- Heston-aware HedgeNet variant (or same HedgeNet with 4 features)
- New experiments 6–8 producing SummaryRows
- Comparison: Heston trained model vs. GBM trained model evaluated on Heston paths

**Lanes**:
- Lane A: implement heston.py, tests
- Lane D: implement heston_features, Heston-aware training
- Lane E: add Heston configs to registry, new experiment scripts
- Lane G: add Heston smoke tests

**What can be parallelized**:
- Lane A (Heston paths) and Lane D (features) are largely independent
- Lane E config additions can happen while Lane A is implementing

**What must be sequential**:
- heston_features requires sample_paths_with_variance from Lane A
- Experiment scripts require both heston.py and heston_features

**Key design note**:
Under Heston there is no simple closed-form BS delta (BS delta uses sigma=sigma_0
as a crude approximation, which ignores vol dynamics). This must be stated clearly
in experiment outputs: "classical baseline uses BS delta with sigma=sqrt(v0)".
This is an approximation, not the true Heston hedge.

---

## Phase 4 — Hedge Universe Study

**Goal**: Compare stock-only vs. stock+option hedge universes.
Corresponds to spec.md Phase D experiments (exp 9–11).

**Experiments to produce**:
9. GBM + call + stock-only vs stock+option
10. Heston + call + stock-only vs stock+option
11. Heston + spread + stock-only vs stock+option

**Entry criteria**:
- Phase 1 complete
- Phase 3 in progress or complete (Heston needed for exp 10–11)

**Exit criteria**:
- `src/hedging/hedge_universe.py` StockOnlyUniverse and StockPlusOptionUniverse implemented
- P&L engine extended to support multi-instrument hedging (coordinate with Lead)
- HedgeNet extended to output 2 hedge ratios for stock+option universe
- New experiments 9–11 producing SummaryRows
- Clear result: does stock+option hedging materially improve CVaR?

**Lanes**:
- Lane B: implement hedge_universe.py and multi-instrument P&L extension
- Lane D: extend HedgeNet for multi-output (hedge ratios per instrument)
- Lane E: add configs, scripts
- Lead: coordinate the multi-instrument P&L contract change

**What must be sequential**:
- Multi-instrument P&L engine (Lane B) must be complete before Lane D can train
- Contract for multi-output HedgeNet must be agreed before Lane D starts

---

## Phase 5 — Ablations, Summaries, and Reporting

**Goal**: Architecture size, feature ablation, objective comparison experiments.
Then generate the final summary table and markdown report.
Corresponds to spec.md Phase E experiments (exp 12–15).

**Experiments to produce**:
12. Architecture size comparison (small/medium/large HedgeNet)
13. Feature ablation (which features matter)
14. Objective comparison (CVaR vs entropic risk loss)
15. Position-aware vs not (current_delta feature on/off)

**Entry criteria**:
- Phases 1–3 complete (at minimum Phase 1 and 2; 3 and 4 are optional for this phase)
- SummaryRows exist in results/raw/ for at least Phase 1–2 experiments

**Exit criteria**:
- All ablation experiments produce SummaryRows
- `src/reporting/tables.py` implemented: build_summary_table produces correct CSV
- `src/reporting/plots.py` implemented: plot_cost_frontier, plot_pnl_distribution
- `src/reporting/report.py` implemented: generate_report writes report.md
- `make reproduce-report` works end-to-end
- README.md updated with full results table and key figures

**Lanes**:
- Lane D: ablation experiments (architecture, features, objective)
- Lane E: ablation experiment scripts
- Lane F: implement reporting pipeline, update README
- Lane G: verify reproduce-report target works cleanly

**What can be parallelized**:
- All four ablation experiments are independent
- Reporting pipeline (Lane F) can be developed while ablations run

**What must be sequential**:
- generate_report requires SummaryRows to exist
- README update happens last

---

## Cross-Phase Risk Register

| Risk | Mitigation |
|---|---|
| pnl.py signature change (Phase 2) | Coordinate with Lead; update all callers atomically |
| Heston simulation quality (Phase 3) | Validate moments against Heston characteristic function |
| _compute_pnl drift from pnl.py (ongoing) | Lane D runs regression test comparing both outputs |
| Config field collision (ongoing) | All new fields require Lead review; never remove without audit |
| SummaryRow schema change (Phase 4+) | Versioned schema; Lane E + Lane F must be notified |

---

## Seed Table (Reserve for Future Experiments)

| Experiment | Seed range |
|---|---|
| Phase 1 baseline | 42–49 |
| Phase 2 multi-payoff | 100–109 |
| Phase 3 Heston | 200–209 |
| Phase 4 hedge universe | 300–309 |
| Phase 5 ablations | 400–419 |
| Repeated-seed studies | 500–599 |

Reserve seeds outside these ranges for future use.
Never reuse a seed between experiments that are meant to be independent.
