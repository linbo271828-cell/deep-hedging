# Paper Notes — Buehler, Gonon, Teichmann, Wood (2019)

**Paper:** Deep Hedging  
**Venue:** Quantitative Finance, 2019  
**arXiv:** 1802.03042  
**Link:** https://arxiv.org/abs/1802.03042

These are the author's own notes, written after reading the paper.
This document is a credibility signal: it shows you read and understood the paper,
not just implemented a vague approximation of it.

Write in your own words. Cite section numbers when helpful. Note what this
project implements vs. what the paper does that is beyond sprint scope.

---

## §1 — What Problem Does the Paper Solve?

*[In 2–4 sentences: what is the gap in existing theory/practice that motivates the paper?
What does classical hedging theory assume that is unrealistic? Why does that matter?]*

*[Hint: think about model misspecification, market frictions, discrete rebalancing,
regulatory capital requirements (CVaR / expected shortfall is now required by Basel III).]*

---

## §2 — The Setup and Notation

*[Summarize: what is the market model? How general is it (semi-martingales)?
What is the hedging objective — what are they minimizing?
What is the time horizon, the payoff function?]*

*[Key notation from the paper:]*
- Z = terminal payoff (e.g., max(S_T - K, 0) for a call)
- delta_t = hedge ratio at time t (what the network outputs)
- P&L of the hedging strategy: ...
- The convex risk measure rho(·): ...

---

## §3 — The Loss Function

*[Write out the loss function from the paper in your own words.
What is minimized? What constraints does the hedging strategy satisfy?
What role does the convex risk measure play?]*

*[Key point: the paper frames hedging as minimizing a convex risk measure of the
residual hedging P&L. This is more general than minimizing expected squared P&L.
Why does the choice of risk measure matter?]*

---

## §4 — The Neural Architecture

*[What network architecture does the paper use? What are the inputs at each timestep?
How does the network handle the sequential nature of hedging (hint: recurrent)?
What activation functions?]*

*[How does this sprint's implementation differ:
- This sprint uses a feedforward (not recurrent) network
- Features are handcrafted: (S_t/S_0, tau_t, current_delta)
- The full paper uses additional market features (implied vol surface, etc.)
- This is honest and should be stated in the README]*

---

## §5 — Key Results from the Paper

*[Summarize the main empirical findings:
- Does the neural hedger outperform delta-hedging under costs? By how much?
- Under stochastic volatility, how does the neural hedger behave vs. BS delta?
- What happens to the learned strategy as costs increase?
- Does the strategy converge to something interpretable (e.g., gamma-vega hedging)?]*

---

## §6 — What This Sprint Implements vs. the Full Paper

*[Be explicit about what is simplified or excluded. This section is important for honesty.]*

| Paper feature | This sprint |
|---|---|
| General semi-martingale markets | GBM only |
| Stochastic volatility (Heston) | GBM only (Heston is a stretch goal) |
| Recurrent neural network (LSTM/GRU) | Feedforward FFN with handcrafted features |
| Multiple assets | Single asset (the call's underlying) |
| Real market data / calibration | Simulated markets only |
| Full implied vol surface as features | S_t/S_0, tau_t, current_delta only |
| Various convex risk measures | CVaR (95%) and entropic risk |
| Regulatory capital constraints | Not implemented |

---

## §7 — Insights and Questions

*[Write down anything surprising, confusing, or particularly elegant from the paper.
What open questions does the paper raise? What would you want to test?]*

---

## §8 — Notation Mapping (Paper → Code)

| Paper symbol | This project | Notes |
|---|---|---|
| Z | `max(S_T - K, 0)` | European call payoff |
| delta_t | output of `HedgeNet.forward()` | hedge ratio in [0,1] |
| rho(·) | `risk_measures.cvar()` | 95% CVaR |
| kappa | `config.cost_rate` | proportional transaction cost |
| lambda | `config.alpha` | CVaR level (0.95) |
| T | `config.T` | maturity in years |
| n | `config.n_steps` | rebalancing steps |
