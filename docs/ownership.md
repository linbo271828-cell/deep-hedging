# Ownership Boundaries — Deep Hedging Lab

This document defines the 7-lane ownership model for agent-team implementation.
Each lane can work independently given the contracts in writeup/architecture.md
and writeup/experiment-contracts.md.

**Lead** is the integration coordinator and sole owner of shared contracts.
**Lanes A–G** are parallel implementation tracks.

---

## Reading This Document

For each lane:
- **Primary files**: the lane writes and owns these files exclusively.
- **Shared read-only**: the lane reads these but must not write to them.
- **High-risk coordination required**: changes to these files need Lead approval.

---

## Lead — Integration and Architecture

**Role**: owns shared contracts, architecture, cross-lane integration, and final report.

**Primary files**:
- `writeup/architecture.md`
- `writeup/experiment-contracts.md`
- `writeup/ownership.md` (this file)
- `writeup/implementation_plan.md`
- `src/experiments/registry.py` — only Lead registers new named experiments
- `CLAUDE.md`
- `README.md`

**Responsibilities**:
- Maintains stable shared contracts (configs, SummaryRow, P&L engine interface).
- Reviews any change to `src/experiments/configs.py` that adds or modifies fields.
- Reviews any change to `src/hedging/pnl.py`.
- Reviews any change to `src/payoffs/european.py` signatures.
- Integrates outputs from all lanes into the final report.

**Coordination rule**: any lane that needs a new config field, a new SummaryRow field,
or a new registry entry must coordinate with Lead before implementing.

---

## Lane A — Market Models

**Role**: implement and test market simulators (GBM complete; Heston Phase C).

**Primary files**:
- `src/models/gbm.py`
- `src/models/heston.py`
- `tests/test_market.py`
- `tests/test_heston.py` (new, Phase C)

**Shared read-only**:
- `src/experiments/configs.py` (reads MarketConfig, HestonConfig — does NOT write)
- `src/utils/seeds.py`

**High-risk coordination required**:
- Changing `GBM.sample_paths` signature → breaks Lane B (hedging), Lane D (neural)
- Adding Heston fields to HestonConfig → coordinate with Lead
- Changing the MarketModel protocol → coordinate with all lanes using it

**Phase A deliverable**: GBM works (already done).
**Phase C deliverable**: Heston.sample_paths + sample_paths_with_variance pass tests.

**Acceptance criteria for Heston**:
- Marginal variance distribution at T matches theoretical for given params.
- Feller condition satisfied with default params.
- Paths are strictly positive (reflected/absorbed at zero if Feller violated).
- Deterministic given seed.

---

## Lane B — Hedging Core

**Role**: owns the P&L engine, transaction costs, and hedge universe.

**Primary files**:
- `src/hedging/pnl.py`
- `src/hedging/transaction_costs.py`
- `src/hedging/hedge_universe.py`
- `tests/test_hedger.py`

**Shared read-only**:
- `src/models/gbm.py` (reads paths)
- `src/payoffs/european.py` (reads call_price for initial option price)
- `src/hedging/baseline.py` (reads classical_delta_pnl for tests)
- `src/experiments/configs.py`

**High-risk coordination required**:
- Changing `hedge_pnl` signature → breaks Lane C (baseline), Lane D (neural), Lane E (runner)
- Changing `proportional_cost` signature → breaks pnl.py
- Adding new cost model types → coordinate with Lead and Lane D

**Phase A deliverable**: hedge_pnl is tested and correct (already done via shim).
**Phase D deliverable**: HedgeUniverse protocol + StockOnlyUniverse + StockPlusOptionUniverse.

---

## Lane C — Payoffs and Baselines

**Role**: owns payoff functions and the classical hedging baselines.

**Primary files**:
- `src/payoffs/european.py`
- `src/payoffs/spreads.py`
- `src/payoffs/straddles.py`
- `src/hedging/baseline.py`
- `tests/test_black_scholes.py`
- `tests/test_payoffs.py` (new, Phase B)

**Shared read-only**:
- `src/hedging/pnl.py` (calls hedge_pnl — does NOT write)
- `src/experiments/configs.py` (reads PayoffConfig)

**High-risk coordination required**:
- Changing `call_price`, `call_delta` signatures → breaks Lane D (neural train uses call_price),
  Lane B (pnl.py), test files
- Adding a new payoff type without registering it in PayoffConfig → coordinate with Lead
- Adding a new classical baseline for spreads/straddles → document "no exact baseline" cases

**Phase B deliverable**: call_spread_payoff, put_spread_payoff, straddle_payoff + tests.

**Rule on classical baselines for complex payoffs**:
If no exact classical baseline exists (e.g., for spreads under Heston), the baseline
function must raise NotImplementedError with a clear explanation, not silently apply
an incorrect approximation.

---

## Lane D — Neural Hedger

**Role**: owns neural network architectures, feature builders, training loop, evaluation.

**Primary files**:
- `src/neural/architectures.py`
- `src/neural/features.py`
- `src/neural/train.py`
- `src/neural/evaluate.py`
- `tests/test_neural_smoke.py` (new)

**Shared read-only**:
- `src/hedging/pnl.py` (mirrors its accounting in _compute_pnl — must stay in sync)
- `src/models/gbm.py` (training calls GBM.sample_paths)
- `src/models/heston.py` (Phase C: Heston-aware features)
- `src/payoffs/european.py` (reads call_price for initial_option_price)
- `src/risk/cvar.py` (differentiable torch version is inlined in train.py)
- `src/experiments/configs.py`

**High-risk coordination required**:
- Changing `_compute_pnl` internal accounting → must stay identical to pnl.py; coordinate with Lane B
- Changing `HedgeNet.forward` input dimension → breaks feature builders; coordinate with Lane C
- Adding new network architectures → register in architectures.py only; do not scatter

**Phase A deliverable**: train() works, evaluate() implemented.
**Phase C deliverable**: heston_features implemented, Heston-aware HedgeNet variant.

**Invariant**: `_compute_pnl` in train.py must implement the same accounting as
`hedge_pnl` in pnl.py. If pnl.py changes, train.py must change. Lane D owns both
sides of this contract for the neural path.

---

## Lane E — Experiment System

**Role**: owns the config system, runner, and new experiment scripts.

**Primary files**:
- `src/experiments/configs.py` (add fields with defaults only; coordinate with Lead to remove)
- `src/experiments/runner.py`
- `src/experiments/summaries.py`
- `experiments/` (new scripts for Phase B onwards)
- `tests/test_runner_smoke.py` (new)

**Shared read-only**:
- `src/experiments/registry.py` (Lead owns this; Lane E may request additions)
- `src/neural/train.py` (runner calls train)
- `src/neural/evaluate.py` (runner calls evaluate)
- `src/hedging/baseline.py` (runner calls classical_delta_pnl)
- `src/utils/io.py` (runner uses save_json)

**High-risk coordination required**:
- Adding new config fields → always add with defaults to preserve backward compatibility
- Removing or renaming config fields → requires Lead approval and updating ALL callers
- Adding new SummaryRow fields → coordinate with Lane F (reporting depends on schema)

**Phase A deliverable**: runner.run_experiment() works end-to-end for Phase A configs.
**Phase B deliverable**: new experiment scripts 10-14 using registry + runner.

---

## Lane F — Reporting and Documentation

**Role**: owns the reporting pipeline, README, writeup, and generated reports.

**Primary files**:
- `src/reporting/tables.py`
- `src/reporting/plots.py`
- `src/reporting/report.py`
- `README.md`
- `writeup/math_reference.md` (user-authored; Lane F structures only)
- `writeup/paper_notes.md` (user-authored; Lane F structures only)

**Shared read-only**:
- `src/experiments/summaries.py` (reads SummaryRow schema)
- `results/` (reads artifacts; does NOT write raw/ or processed/)

**High-risk coordination required**:
- Changing the SummaryRow schema → coordinate with Lead and Lane E
- Changing the artifact layout → coordinate with Lead and Lane G

**Phase E deliverable**:
- `build_summary_table(rows)` generates a pandas DataFrame
- `plot_cost_frontier(rows, path)` generates the headline figure
- `generate_report(results_dir, output_path)` generates the markdown report

**Plotting standards**:
- 150 dpi minimum
- (8, 5) single-panel, (12, 5) side-by-side
- Always: axis labels, title, legend when >1 series
- Matplotlib default palette (no seaborn unless significant savings)

---

## Lane G — QA and Reproducibility

**Role**: owns the test suite integrity, lint/typecheck, Makefile, and reproduce pipeline.

**Primary files**:
- All `tests/test_*.py` (coordinates additions with lane owners)
- `Makefile`
- `pyproject.toml` (tool configuration)
- `requirements.txt`
- `.gitignore`

**Shared read-only**: everything

**High-risk coordination required**:
- Changing `pyproject.toml` tool config → affects all lanes' lint/typecheck
- Adding new test files for modules owned by other lanes → coordinate with that lane
- Changing reproduce targets → coordinate with Lead

**Responsibilities**:
- All tests pass at all times on main branch.
- `make lint` and `make typecheck` are clean at all times.
- `make reproduce-core` runs end-to-end without error.
- CI configuration (if/when added).
- Smoke tests for each newly implemented module (coordinate with owning lane).

---

## Coordination Matrix

| Change | Owner | Must notify |
|---|---|---|
| New config field | Lane E (add with default) | Lead, Lane G |
| Remove/rename config field | Lead only | All lanes |
| Change hedge_pnl signature | Lead + Lane B | Lane C, Lane D, Lane E |
| Change _compute_pnl accounting | Lane D | Lane B (must match pnl.py) |
| Change call_price/call_delta signature | Lead + Lane C | Lane B, Lane D, tests |
| Add new payoff type | Lane C | Lead (registry entry), Lane E (config update) |
| Add Heston support | Lane A + Lead | Lane D (features), Lane E (configs) |
| Add new SummaryRow field | Lead + Lane E | Lane F (tables/plots) |
| Add new registry entry | Lead only | Lane E (runner), Lane G (tests) |
| Change artifact layout | Lead + Lane E | Lane F (reporting), Lane G (repro) |
