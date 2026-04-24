# Math Reference — Deep Hedging Sprint

This document is written by the project author in their own words.
It serves two purposes:
1. A personal study reference — forces you to reconstruct the math, not just copy it.
2. A credibility signal for quant-firm readers who will skim the repo.

Write each section after you have derived it on paper. Paraphrase Hull/Shreve/Wilmott
in your own language. Do not copy formulas without understanding them.

---

## §1 — Geometric Brownian Motion

### The SDE

*[Write here: the GBM SDE, what each term means, why it is a reasonable model for stock prices.]*

```
dS_t = mu * S_t * dt + sigma * S_t * dW_t
```

### Ito's Lemma

*[Derive Ito's lemma for f(S_t) = log(S_t). Show each step. State the result for a general
smooth f(S_t, t).]*

*[Key insight: why does the second-derivative term appear? What is the quadratic variation
of a Wiener process?]*

### Exact Solution for S_t

*[Starting from Ito's lemma applied to log(S), derive the exact formula:]*

```
S_t = S_0 * exp( (mu - sigma^2/2) * t + sigma * W_t )
```

*[Why is the drift term (mu - sigma^2/2)*t and not mu*t? What does this mean for
log-returns vs. arithmetic returns?]*

### Moments of log(S_T / S_0)

*[Derive:]*
```
E[log(S_T / S_0)]   = (mu - sigma^2/2) * T
Var[log(S_T / S_0)] = sigma^2 * T
```

*[These are exactly what the GBM sanity tests verify. Connect the math to the code.]*

### The Exact Simulation Scheme

*[Explain why the project uses the exact exponential formula rather than Euler-Maruyama.
What error does Euler-Maruyama introduce? Why does it matter for option pricing?]*

---

## §2 — Black-Scholes: The Hedging Argument

### Setup

*[Define: European call with strike K, maturity T, underlying S_t.
Define: risk-free bond with rate r. Frictionless market. GBM dynamics for S.]*

### The Replication Portfolio

*[Describe the delta-hedging portfolio: hold Delta shares + cash account.
What does "self-financing" mean? Why must the portfolio value satisfy V = Delta*S + B?]*

### Deriving the Black-Scholes PDE

*[Apply Ito's lemma to V(S_t, t). Set up the replication argument (the portfolio is
riskless, so it must earn the risk-free rate). Derive the PDE:]*

```
dV/dt + (1/2)*sigma^2*S^2*(d^2V/dS^2) + r*S*(dV/dS) - r*V = 0
```

*[State the boundary condition for a European call: V(S, T) = max(S-K, 0).]*

### Risk-Neutral Pricing

*[State (without full proof) that the BS formula can also be derived by risk-neutral pricing:
discounting the expected payoff under the risk-neutral measure Q where mu is replaced by r.
This connects to Monte Carlo pricing — what we verify in test_mc_price_matches_closed_form.]*

### The Closed-Form Solution

*[Write out the BS call price formula:]*

```
C = S * N(d1) - K * exp(-r*tau) * N(d2)

d1 = (log(S/K) + (r + sigma^2/2)*tau) / (sigma*sqrt(tau))
d2 = d1 - sigma*sqrt(tau)
```

*[Interpret each term: S*N(d1) = expected value of stock if called; K*exp(-r*tau)*N(d2) =
discounted expected cost of exercise. Explain N(d1) as the hedge ratio Delta.]*

### Put-Call Parity

*[Derive from no-arbitrage:]*

```
C - P = S - K * exp(-r*tau)
```

*[This is tested in test_put_call_parity. Explain the arbitrage argument if violated.]*

---

## §3 — The Greeks

For each Greek, write the formula, derive it or state the financial interpretation,
and note any numerical edge cases handled in the code.

### Delta — dC/dS = N(d1)

*[Derive by differentiating the BS formula. Verify the d1-partial cancellation.
Interpret: Delta shares of stock neutralize first-order price risk.
Edge cases: Delta → 1 as S → ∞ (deep ITM), Delta → 0 as S → 0 (deep OTM).]*

### Gamma — d^2C/dS^2 = phi(d1) / (S*sigma*sqrt(tau))

*[Derive. Interpret: rate of change of delta; measures convexity exposure.
Gamma is always positive for calls and puts.
Edge case: Gamma → ∞ as tau → 0 for ATM options.]*

### Vega — dC/d(sigma) = S * phi(d1) * sqrt(tau)

*[Derive. Interpret: sensitivity to implied volatility.
Note: Vega is the same for calls and puts (by put-call parity, since parity is linear in C).]*

### Theta — dC/dt (time decay)

*[Write out the formula. Interpret: usually negative for calls — the option loses value
as time passes (for long option holders).]*

### Rho — dC/dr

*[Write out the formula. Usually less important for short-dated options.]*

---

## §4 — Risk-Neutral Measure and Monte Carlo Pricing

### The Risk-Neutral Measure Q

*[State (intuitively): under Q, all assets grow at the risk-free rate r.
In GBM: set mu = r. The discounted stock price is a martingale under Q.]*

*[Why does this matter? Option prices = E^Q[discounted payoff]. This is how the MC
test works: simulate GBM with mu = r, average discounted payoffs, compare to BS formula.]*

### Monte Carlo Convergence

*[State the central limit theorem result: MC price error is O(1/sqrt(N)).
With N = 200k paths, the standard error is roughly sigma_payoff / sqrt(200000).
The test uses a 3-sigma bound — explain why this is sufficient.]*

---

## §5 — Transaction Costs and Hedging Under Frictions

### The Classical Hedger with Costs

*[Describe the discrete-time hedging strategy:
- At each rebalancing step, compute BS delta.
- Trade to reach the target delta.
- Pay |trade| * S * cost_rate in costs.

Write out the P&L formula. What is the expected P&L without costs (should be ~0)?
How do costs change the P&L distribution (mean becomes negative, variance increases)?]*

### Why Classical Hedging Is Suboptimal Under Costs

*[Gamma-cost tradeoff: hedging reduces variance (Gamma exposure) but costs money.
The optimal strategy under costs trades less frequently, accepting higher variance
in exchange for lower costs. Black-Scholes delta-hedging cannot express this tradeoff —
it rebalances to the formula, regardless of cost.]*

---

## §6 — Risk Measures

### CVaR (Conditional Value at Risk / Expected Shortfall)

*[Define: at confidence level alpha, CVaR is the expected loss in the worst (1-alpha)
fraction of outcomes.]*

```
CVaR_alpha(X) = E[-X | -X >= VaR_alpha(X)]
```

*[Why CVaR and not VaR? CVaR is subadditive (diversification reduces risk).
VaR is not subadditive. For a coherent risk measure, subadditivity is required.]*

*[How CVaR is used as a training loss: minimize CVaR of terminal P&L over the training
distribution. The network learns to reduce tail losses, not just expected losses.]*

### Entropic Risk Measure

*[Define:]*

```
rho_lambda(X) = (1/lambda) * log( E[exp(-lambda * X)] )
```

*[Interpret: exponential utility with risk-aversion parameter lambda.
lambda → 0 recovers the expected loss. lambda → ∞ recovers worst-case loss.]*

*[Note: both CVaR and entropic risk are convex risk measures in the sense of Föllmer-Schied.
This ensures the training loss is (at least) locally well-behaved.]*

---

## §7 — References

- Hull, J. *Options, Futures, and Other Derivatives*, Ch. 13–17.
- Wilmott, P. *Paul Wilmott Introduces Quantitative Finance*, Ch. 2–5.
- Shreve, S. *Stochastic Calculus for Finance II*, Ch. 4 (for rigorous treatment).
- Föllmer, H., Schied, A. *Stochastic Finance*, Ch. 4 (risk measures).
- Buehler et al. *Deep Hedging*, Quantitative Finance, 2019 (arXiv: 1802.03042).
