# Experiment Contracts — Deep Hedging Lab

This document specifies the canonical config and artifact schema for all experiments.
These contracts are the ground truth for:
- What inputs each experiment requires
- What outputs it produces
- How results are stored and aggregated

---

## Config Hierarchy

All configs are frozen dataclasses in `src/experiments/configs.py`.
They compose into `ExperimentConfig` which is the single input to the runner.

```
ExperimentConfig
├── name: str
├── market: MarketConfig | HestonConfig
├── payoff: PayoffConfig
├── hedge_universe: HedgeUniverseConfig
├── training: TrainingConfig
└── evaluation: EvaluationConfig
```

---

## MarketConfig

Controls GBM path generation.

| Field | Type | Default | Description |
|---|---|---|---|
| s0 | float | 100.0 | Initial stock price |
| mu | float | 0.05 | Physical drift (not used in risk-neutral training) |
| r | float | 0.02 | Risk-free rate (continuous compounding) |
| sigma | float | 0.20 | GBM volatility |
| T | float | 0.5 | Option maturity in years |
| n_steps | int | 50 | Number of discrete rebalancing steps |
| k | float | 100.0 | Strike price (kept here for convenience; also in PayoffConfig) |

**Risk-neutral convention**: training always uses `mu = r`. The `mu` field is for
physical-measure simulation only (e.g., moment validation experiments).

---

## HestonConfig

Controls Heston path generation. Stub — not yet active.

| Field | Type | Default | Description |
|---|---|---|---|
| s0 | float | 100.0 | Initial stock price |
| r | float | 0.02 | Risk-free rate |
| v0 | float | 0.04 | Initial variance (sigma_0 = 20%) |
| kappa | float | 2.0 | Mean-reversion speed |
| theta | float | 0.04 | Long-run variance |
| xi | float | 0.3 | Vol-of-vol |
| rho | float | -0.7 | Stock-variance correlation |
| T | float | 0.5 | Maturity |
| n_steps | int | 50 | Rebalancing steps |

Feller condition: `2 * kappa * theta >= xi**2`. Must hold for well-posed simulation.
With defaults: 2*2.0*0.04 = 0.16 >= 0.09 = 0.3^2. ✓

---

## PayoffConfig

Defines the option being hedged (the short position whose P&L we evaluate).

| Field | Type | Default | Description |
|---|---|---|---|
| payoff_type | Literal | "call" | One of: "call", "put", "call_spread", "straddle" |
| k | float | 100.0 | Primary strike |
| k_hi | float | 110.0 | Upper strike (spreads only; ignored for others) |

---

## HedgeUniverseConfig

Defines which instruments the hedger can trade.

| Field | Type | Default | Description |
|---|---|---|---|
| universe_type | Literal | "stock_only" | "stock_only" or "stock_plus_option" |
| hedge_option_strike | float | 100.0 | Liquid option strike (stock_plus_option only) |
| hedge_option_maturity | float | 0.5 | Liquid option maturity |

---

## TrainingConfig

Neural hedger training hyperparameters.

| Field | Type | Default | Description |
|---|---|---|---|
| n_paths | int | 50_000 | Paths per training epoch |
| n_epochs | int | 200 | Training epochs |
| lr | float | 1e-3 | Adam learning rate |
| hidden_dim | int | 64 | Hidden units per layer |
| n_layers | int | 4 | Number of hidden layers |
| seed | int | 42 | Master seed (PyTorch + training path seed) |
| cost_rate | float | 0.0 | Proportional cost rate (e.g. 0.0005 = 5 bps) |
| alpha | float | 0.95 | CVaR confidence level |

**Seed convention**: training paths use `derive_seed(seed, "train")`. Evaluation paths
use `derive_seed(seed, "eval")`. This ensures independence while remaining reproducible.

---

## EvaluationConfig

Out-of-sample evaluation settings.

| Field | Type | Default | Description |
|---|---|---|---|
| n_paths | int | 100_000 | Number of evaluation paths |
| seed_offset | int | 1000 | Added to training seed to get eval seed |
| alpha | float | 0.95 | CVaR level |
| lambda_ | float | 1.0 | Entropic risk lambda |

**Seed convention**: eval seed = `derive_seed(training.seed + seed_offset, "eval")`.
The offset ensures eval paths are independent of training paths for ALL experiments.

---

## ExperimentConfig

Umbrella config. The single input to `run_experiment(name)`.

| Field | Type | Description |
|---|---|---|
| name | str | Unique experiment name (key in registry) |
| market | MarketConfig | GBM parameters |
| payoff | PayoffConfig | Option contract |
| hedge_universe | HedgeUniverseConfig | Tradable instruments |
| training | TrainingConfig | Neural hedger training |
| evaluation | EvaluationConfig | Evaluation settings |

**Rule**: every experiment in `src/experiments/registry.py` maps a name to an
ExperimentConfig. No magic constants outside of configs.

---

## SummaryRow

The canonical output unit of a completed experiment run.

```python
@dataclass
class SummaryRow:
    experiment_name: str
    strategy: str          # "neural" | "classical"
    seed: int
    cost_rate: float
    mean_pnl: float
    std_pnl: float
    cvar_95: float
    entropic_risk: float
    expected_cost: float
    turnover: float
    n_paths: int
    market_model: str      # "gbm" | "heston"
    payoff_type: str       # "call" | "put" | "call_spread" | "straddle"
    hedge_universe: str    # "stock_only" | "stock_plus_option"
    n_epochs: int | None
    hidden_dim: int | None
    n_layers: int | None
```

**Rules**:
- One SummaryRow per (experiment_name, strategy, seed).
- Use None for fields not applicable to a strategy (e.g. n_epochs for classical).
- SummaryRow.to_dict() serializes to JSON/CSV.

---

## Artifact Layout

```
results/
  raw/
    <experiment_name>/
      config.json          # ExperimentConfig serialized
      neural_summary.json  # SummaryRow for neural strategy
      classical_summary.json  # SummaryRow for classical strategy
      loss_history.json    # list of per-epoch CVaR (neural only)
      model_checkpoint.pt  # trained model weights (neural only)
  processed/
    summary.csv            # all SummaryRows concatenated
    summary.json           # same, as JSON
  reports/
    report.md              # generated markdown report
    *.png                  # figures embedded in report
```

**Rules**:
- Raw artifacts are written by `src/experiments/runner.py`.
- Processed artifacts are written by `src/reporting/tables.py`.
- Reports are written by `src/reporting/report.py`.
- Plots are saved by `src/reporting/plots.py`.
- Nothing in `experiments/` writes to results/ directly; it goes through runner.

---

## Win Criterion

A learned hedge is meaningfully better than the classical baseline if and only if:
1. It achieves lower CVaR or lower expected cost (or both) out-of-sample.
2. The improvement is stable across at least 3 different random seeds.
3. The improvement is present at an economically meaningful cost level (>= 5 bps).

Do not claim a win if the neural hedger only improves on the training seed.

---

## Standard Evaluation Metrics

Every SummaryRow must contain all of these (no None for implemented strategies):

| Metric | Definition | Good direction |
|---|---|---|
| mean_pnl | E[P&L] | Higher (closer to 0 for a well-hedged book) |
| std_pnl | std(P&L) | Lower |
| cvar_95 | CVaR at 95% | Lower |
| entropic_risk | (1/lambda)*log(E[exp(-lambda*P&L)]) | Lower |
| expected_cost | E[total transaction cost per path] | Context-dependent |
| turnover | E[sum of |delta_changes|] | Context-dependent |
