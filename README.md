# Deep Hedging Lab

A friction-aware hedging research suite that compares **classical Black-Scholes delta
hedging** against **learned neural hedging policies** across market models, option payoffs,
hedge universes, and transaction-cost regimes.

Inspired by: Buehler, Gonon, Teichmann, Wood — *Deep Hedging* (Quantitative Finance 2019;
arXiv 1802.03042).

The project ships as three coordinated pieces:

1. **A reproducible Python research library** (`src/`) with a P&L engine, market
   simulators, closed-form pricers, a PyTorch neural hedger, and named experiment
   configs.
2. **A FastAPI backend** (`services/sim/`) that exposes the research library over HTTP
   so experiments can be launched, polled, and visualised from a browser.
3. **A Next.js 15 / React 19 frontend** (`apps/web/`) — a dark, editorial-feeling lab UI
   with an audience-mode system (Beginner / Guided / Expert), preset pickers,
   metrics tables, P&L histograms, and an interactive 3-mode Plotly 3D hedge surface.

---

## What it does

When you sell a European call option, you hedge by dynamically holding a position in
the underlying stock. The classical prescription (Black-Scholes) says: hold
`Δ = N(d₁)` shares and rebalance continuously. Under idealised assumptions this is
optimal.

**Under transaction costs it is not.** Every trade costs money; rebalancing to the
exact delta at every step eats P&L. The question this lab explores is:

> Can a small neural network, trained end-to-end on simulated P&L with a **CVaR**
> objective, learn a hedging policy that trades off variance reduction against turnover
> more intelligently than the formula?

The answer is *yes, under some cost regimes* — and the lab is built to **let you poke
at exactly when and how** that holds, across four payoff families (call, put, bull call
spread, straddle) and two hedge universes (stock-only, stock + one liquid option).

---

## How to use it (the UI flow)

1. On first visit, the app asks you to pick an **audience mode**:
   - **Beginner** — plain-English explanations, glossary pills, and auto-open
     `ExplainBox` cards.
   - **Guided** — concise technical framing on every chart.
   - **Expert** — minimal explanatory text; tables and the surface do the talking.
2. The landing page shows a **preset grid**. Each preset fixes a (market model, payoff,
   hedge universe, cost rate) 4-tuple and has a payoff thumbnail, a cost pill, and a
   short description. Clicking selects; a detail panel below shows the formula, Δ
   range, benchmark, and cost. Clicking **Launch experiment** kicks off a run.
3. The results page polls run status, then renders:
   - A **run-info strip** (market / payoff / universe / cost / eval paths / architecture).
   - A **contract structure** card (payoff, Δ range, benchmark, profile).
   - A **metrics table** with Neural vs Classical for CVaR₉₅, entropic risk, mean P&L,
     std P&L, expected cost, and turnover — with per-row winner/loser coloring and a
     percent-delta column.
   - A **P&L distribution** image (20k out-of-sample paths, mean + 5th percentile
     overlaid).
   - A **3D hedge surface** with a segmented toggle: Classical Δ (viridis), Neural
     (viridis), or Neural − Classical (diverging RdBu), all from backend-computed
     grid data with `prev_delta = 0`.

Everything is instrumented with mode-conditional copy so the same run page looks
appropriate for a student and for a quant interviewer.

---

## Features I'm most proud of

- **Modular `src/` library** with locked shared contracts (see `CLAUDE.md`). The P&L
  engine, pricers, risk measures, and neural hedger are pure-Python and unit-tested;
  the backend and experiments are thin wrappers. 134 tests pass in under 4 seconds.
- **Multi-payoff dispatch.** The same training loop and P&L engine handle call, put,
  call spread, and straddle via a payoff-aware delta function and payoff callable.
- **Stock + option hedge universe.** An optional second instrument (a liquid ATM
  call) can be added; the P&L engine generalises to multi-instrument positions.
- **Surface endpoint.** `GET /api/runs/{id}/surface` returns precomputed `[n_τ][n_s]`
  grids for BS delta, neural delta (at `prev_delta = 0`), and their difference, so the
  frontend can render a live 3D Plotly visual without re-running the network.
- **Audience-mode UI.** A `ModeProvider` + `body[data-mode="…"]` attribute drives
  CSS and React-conditional content across the whole app, with localStorage
  persistence and a first-visit picker modal.
- **Dark, editorial design system.** Fraunces display + Inter body + JetBrains Mono
  for data, with an ink palette, payoff-family hues (call=blue, put=purple,
  spread=cyan, straddle=amber), and a diverging cold/warm accent for difference
  surfaces.

---

## Running it locally

### Prerequisites

- Python 3.11+
- Node.js 18+ (for the frontend)
- `make`

### One-shot install

```bash
# From the repo root:
make install          # Python venv + requirements + editable src/
make install-web      # npm install inside apps/web
```

### Verify the numerics

```bash
make test             # 134 tests, < 4s
```

### Run the full stack in development

Open two terminals:

```bash
# Terminal 1 — FastAPI backend on :8000
make run-backend

# Terminal 2 — Next.js frontend on :3000
make run-frontend
```

Then open <http://localhost:3000>. The frontend reads `NEXT_PUBLIC_API_URL` (defaults
to `http://localhost:8000`).

### Reproduce the v0.1 CLI plots from scratch

```bash
make reproduce-core   # runs experiments 01–05, saves plots/ + results/raw/
```

This regenerates the cost-frontier figure and the individual-experiment plots used in
the v0.1 README; it is the offline, library-only path and does not need the web app.

---

## Project layout

```
deep-hedging/
├── src/                     # Research library (Python)
│   ├── models/              # Market simulators (GBM; Heston stub)
│   ├── payoffs/             # European call/put, spreads, straddles
│   ├── hedging/             # P&L engine, transaction costs, baseline, hedge universe
│   ├── risk/                # CVaR, entropic risk
│   ├── neural/              # HedgeNet FFN, training loop, evaluation
│   ├── experiments/         # Configs, registry, runner, summaries
│   ├── reporting/           # Tables, plots, report generation
│   └── utils/               # Seeds, I/O, stats
│
├── services/sim/            # FastAPI backend (presets, runs, results, plot, surface)
│
├── apps/web/                # Next.js 15 frontend (React 19, TypeScript, Plotly)
│   ├── app/                 # App Router pages (landing, runs/[id], layout)
│   ├── components/          # Nav, Footer, ModeSelector, ModeSwitcher, ExplainBox,
│   │                        # HedgeSurface, HeroSurfacePreview, PayoffShape, CostFrontier
│   └── lib/                 # api.ts (typed client), mode.tsx (context), families.ts
│
├── experiments/             # Named CLI entrypoints (01_validate_market … 05_cost_frontier)
├── tests/                   # 134 pytest numerical + smoke tests
├── results/                 # Generated artifacts (gitignored)
├── writeup/                 # architecture.md, math_reference.md, paper_notes.md, …
├── CLAUDE.md                # Primary operating guide for AI-assisted editing
├── spec.md                  # Full product + engineering spec
├── README.md                # This file
├── PROMPT_LOG.md            # AI tools, process, important prompts
└── REFLECTION.md            # Course reflection (written by hand)
```

---

## Secrets handling

**There are no secrets in this project.** The backend is a local FastAPI server, the
Python library is fully deterministic (no external API calls), and the frontend
reads a single non-secret environment variable:

- `NEXT_PUBLIC_API_URL` — URL of the backend. Defaults to `http://localhost:8000`. For
  a Vercel deployment, set this in the Vercel dashboard pointing at the deployed
  backend.

No API keys, no credentials, no third-party services. Deployment follows the standard
Vercel (frontend) + Render/Railway (backend) split; the backend URL is the only thing
the frontend needs to know.

The repo's `.gitignore` excludes `.env*`, `.venv/`, `node_modules/`, `results/raw/`,
and `__pycache__/` so nothing sensitive or machine-specific is tracked.

---

## Honest limitations

This is a **portfolio-scoped v1**, not a full reproduction of Buehler (2019):

- **Simulated markets only.** No calibration to real option chains. Results may not
  transfer to real markets.
- **GBM only in v1.** Heston is stubbed but not wired; under GBM, Black-Scholes is the
  true frictionless optimum, so the neural hedger's edge is limited to cost-awareness
  rather than model misspecification.
- **Feedforward only.** The paper uses RNN-based hedgers for path-dependent policies.
  This project uses a 4×64 FFN with handcrafted features.
- **Proportional costs only.** No spread costs, no market impact, no slippage.
- **Single seed per run.** The UI runs one seed; the CLI experiments sweep seeds and
  report standard errors.

These are called out loudly in the UI (in the mode banners and footer text) and in the
writeup.

---

## Dependencies

**Python:** NumPy, SciPy, PyTorch, Matplotlib, pandas, FastAPI, uvicorn, pytest, ruff,
mypy, black. See `requirements.txt` and `pyproject.toml`.

**Frontend:** Next.js 15, React 19, TypeScript, Plotly.js, react-plotly.js. See
`apps/web/package.json`.

Fonts (Fraunces, Inter, JetBrains Mono) are self-hosted via `next/font/google` and
cached at build time — no runtime network call to Google is made by the app shell.

---

## Background reading

- `writeup/math_reference.md` — Itō lemma, GBM dynamics, BS PDE, the Greeks, CVaR,
  entropic risk.
- `writeup/paper_notes.md` — annotation of Buehler (2019).
- `writeup/architecture.md` — module-boundary decisions and the seven ownership lanes.
- `writeup/experiment-contracts.md` — config + artifact schema.

---

## License

MIT
