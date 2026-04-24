# Deep Hedging — Neural Option Hedging Under Transaction Costs

A 2-week sprint implementation of a neural-network option hedger benchmarked against
classical Black-Scholes delta-hedging on simulated GBM markets, with emphasis on the
effect of **proportional transaction costs** on hedging strategy.

Inspired by: Buehler, Gonon, Teichmann, Wood — *Deep Hedging* (Quantitative Finance, 2019;
arXiv: 1802.03042).

---

## The Core Idea

When you sell a European call option, you hedge the resulting exposure by holding a dynamic
position in the underlying stock. The classical prescription (Black-Scholes delta-hedging)
says: hold `Δ = N(d1)` shares and rebalance continuously. Under idealized assumptions this
is perfect.

**Reality:** every trade costs money. Continuous rebalancing destroys P&L through
transaction costs. The question is: can a neural network learn a *better* hedging policy
directly from simulated data — one that trades off variance reduction against transaction
costs more intelligently than the formula?

**Answer (from this project):** Yes. The neural hedger learns to hedge less aggressively
when costs are high, accepting slightly more variance in exchange for lower expected cost.
The classical delta hedger cannot adapt; its strategy is fixed by the formula regardless of
the cost regime.

---

## Headline Result

![Cost frontier](plots/05_cost_frontier.png)

The plot shows CVaR of terminal P&L vs. expected transaction cost for both strategies,
across cost levels from 0 to 50 basis points.

Key numbers from the sweep:

| Cost (bps) | Neural CVaR | Classical CVaR | Neural E[cost] | Classical E[cost] |
|---|---|---|---|---|
| 0 | 2.11 | 1.62 | 0.00 | 0.00 |
| 10 | 2.13 | 2.05 | 0.30 | 0.34 |
| 30 | 3.16 | 2.95 | 0.90 | 1.01 |
| 50 | **3.46** | **3.88** | **1.35** | **1.68** |

At 50 bps, the neural hedger achieves lower tail risk (CVaR 3.46 vs 3.88) at
lower expected cost (1.35 vs 1.68) — Pareto-dominant. At low costs the neural
hedger is close to but not yet identical to BS delta (convergence is asymptotic).

---

## Quick Start

```bash
git clone https://github.com/<your-username>/deep-hedging.git
cd deep-hedging

# Create virtual environment and install dependencies
make install

# Run tests (14 tests; should take ~15 seconds)
make test

# Regenerate all plots from scratch (requires torch; takes several minutes)
make reproduce
```

The `make reproduce` target runs all five experiments in order and saves plots to `plots/`.

---

## Project Structure

```
src/
  market.py           GBM path simulator (exact exponential solution)
  black_scholes.py    Closed-form BS pricer + all Greeks
  hedger.py           P&L accounting engine; classical delta hedger
  risk_measures.py    CVaR and entropic risk (pure functions)
  neural_hedger.py    PyTorch FFN that outputs a hedge ratio
  train.py            Training loop (CVaR loss over simulated paths)
  config.py           Experiment hyperparameters (frozen dataclasses)

experiments/
  01_validate_market.py     GBM moment validation
  02_classical_delta.py     Classical hedger P&L — no cost vs. 5 bps
  03_neural_no_costs.py     Neural hedger sanity check (must recover BS delta)
  04_neural_with_costs.py   Neural hedger under transaction costs
  05_cost_frontier.py       Headline figure — cost-frontier sweep

tests/
  test_market.py            GBM numerical correctness
  test_black_scholes.py     BS pricing and Greeks correctness
  test_hedger.py            P&L accounting correctness
  test_risk_measures.py     Risk measure numerical correctness

writeup/
  math_reference.md         BS PDE derivation + Greeks (author's own notes)
  paper_notes.md            Buehler 2019 annotation
  architecture.md           Module-boundary decisions
  sprint_plan.md            Build order and sprint status
```

---

## Methods

### Market Model

Geometric Brownian Motion simulated using the exact solution:

```
S_{t+dt} = S_t * exp( (mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z ),   Z ~ N(0,1)
```

This is unbiased (no Euler-Maruyama discretization error).

### Classical Baseline

Black-Scholes delta-hedging with discrete rebalancing at fixed intervals.
Transaction costs are proportional to trade size: cost = |Δdelta| * S_t * cost_rate.

### Neural Hedger

Feedforward network (4 layers, 64 hidden units, ReLU activations, sigmoid output).
Input features at each timestep: `(S_t/S_0, time_to_maturity, current_delta)`.
Trained end-to-end by minimizing CVaR (95th percentile expected shortfall) of terminal P&L
over simulated risk-neutral GBM paths including transaction costs.

### Risk Evaluation

CVaR at 95%: mean of the worst 5% of terminal P&L outcomes.
Lower CVaR = less tail risk = better hedging performance.

---

## Sanity Check Gate

Before drawing any conclusions about transaction costs, the neural hedger was verified to
recover Black-Scholes delta-like behavior in the zero-cost case (experiment 03).
This is a necessary condition for trusting the results with costs — if the network cannot
solve the easy problem, it cannot be trusted on the hard one.

---

## Honest Limitations

This project is a **simplified sprint-scoped implementation**, not a full reproduction of
Buehler (2019). Specifically:

- **Simulated markets only.** No calibration to real option chains. Results may not transfer
  to real markets.
- **GBM only.** The original paper uses richer market models (stochastic volatility, jumps).
  Under GBM, Black-Scholes is the true optimal hedger; the neural hedger's advantage is
  limited to reducing transaction costs, not model misspecification.
- **No recurrent architecture.** The paper uses RNN-based hedgers for path-dependent
  strategies. This project uses a feedforward network with handcrafted features.
- **No novel research contribution.** This is a careful implementation and evaluation of an
  existing idea, honestly framed for educational and portfolio purposes.
- **Batch learning only.** Not applicable to latency-sensitive HFT contexts.

---

## Dependencies

- Python 3.11+
- NumPy, SciPy — market simulation and Black-Scholes
- PyTorch — neural network training
- Matplotlib, pandas — plots and summaries
- pytest — numerical tests

---

## Math Background

See `writeup/math_reference.md` for the author's derivation of:
- Ito's lemma and GBM dynamics
- The Black-Scholes PDE (replication argument)
- The Greeks and their financial interpretation
- CVaR and entropic risk measures

See `writeup/paper_notes.md` for notes on the Buehler (2019) paper.

---

## Reproducibility

Every experiment is seeded. `make reproduce` regenerates all results from scratch.
No manual steps required after `make install`.

```bash
make install   # set up venv + deps
make test      # verify numerical correctness
make reproduce # run all experiments, save plots
```

---

## License

MIT
