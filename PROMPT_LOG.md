# PROMPT_LOG.md — Deep Hedging Lab

A chronological record of the AI-assisted development process for Deep Hedging Lab,
including the tools used, who wrote what, and the non-trivial prompts that shaped the
work.

---

## AI tools and models used

- **Claude Code (CLI)** as the primary coding agent — interactive terminal workflow
  with direct access to the repo (Read / Edit / Write / Bash / Grep / Glob tool calls).
  - Model: **Claude Opus 4.7** for most of the build; the earlier Python-library
    phases were done on the previous Opus/Sonnet generation. I stayed on Opus for the
    larger refactors (modularisation, multi-payoff dispatch, stock+option universe,
    frontend design integration) because the reasoning budget matters most there.
- **Claude Design** (the prototype generator) to produce a dark editorial design
  system as raw HTML/CSS/JSX. I then fed those files back into Claude Code to
  integrate into the live Next.js app.
- **Git worktrees** — the Claude Code session ran inside a feature worktree
  (`.claude/worktrees/flamboyant-swartz-9bbdf4`) so risky edits stayed isolated
  from `main` until I approved them.
- **FastAPI's `/docs`** (Swagger UI) to manually exercise each backend endpoint
  while the frontend was wired up.
- **pytest** locally for the 134-test suite and **`npm run build`** for the Next.js
  type + production-build check. These were the ground-truth pass/fail gates after
  every major edit.

I did **not** use Cursor, Copilot, or any browser chat. The entire project was
developed in a terminal-first agentic loop, because that's the workflow that feels
most like real engineering to me.

---

## Development process, start to finish

### Phase 0 — Spec + math (me, mostly by hand)

I started by reading the Buehler (2019) paper and writing my own notes
(`writeup/paper_notes.md`) and math derivations (`writeup/math_reference.md`) for
Itō's lemma, GBM dynamics, the BS PDE, and the Greeks. I also wrote `spec.md` by
hand to lock scope (what was in, what was out). Claude was asked to proofread but
did not draft this content — it was important that the math was mine.

### Phase 1 — v0.1 library + CLI experiments (mostly Claude)

One large "build deep hedging v0.1" prompt produced the initial flat `src/*.py`
layout, experiments 01–05, and 34 tests. I reviewed the outputs against the spec,
fixed several numerical issues (most notably seed discipline: the agent initially
used `np.random.seed()` globally; I made it use `np.random.default_rng(seed)` per
call, and documented this in `CLAUDE.md`). Commit: `8fa50b1`.

### Phase 2 — modular refactor + FastAPI + Next.js scaffold (Claude, me reviewing)

Next, I asked Claude to migrate the flat `src/*.py` files into
`src/{models,payoffs,hedging,risk,neural,experiments,reporting,utils}/` with
backward-compatible shims, and to add a FastAPI backend + a Next.js 15 frontend
scaffold. Commit: `28f1f84` (the big one: +9,438 / −1,482).
I watched the shim strategy carefully — I wanted old imports to keep working without
silent behaviour change — and spot-checked the new contracts against the locked
signatures in `CLAUDE.md`.

### Phase 3 — multi-payoff + surface endpoint + stock+option (Claude with tight prompts)

- **Multi-payoff dispatch.** I asked Claude to generalise the classical delta function
  and P&L engine to accept arbitrary payoffs (call / put / call_spread / straddle)
  rather than hardcoding calls. This surfaced a real bug — the neural hedge network
  features were only valid for calls — and the fix involved making features
  payoff-aware. Commit: `131e833`.
- **Surface endpoint.** Backend computation of the three grids (BS, neural, diff) at
  `prev_delta = 0`, then shipped to the frontend as JSON. Claude's first pass had the
  grids transposed; I caught it by comparing to the analytical `N(d₁)` at S=K, τ=0.5.
- **Stock + option hedge universe.** A second liquid ATM call added as a hedge
  instrument with its own delta and cost stream. Commit: `42c101e`.

### Phase 4 — audience modes (Claude, long conversation)

I wanted the frontend to gracefully serve three audiences — a CS classmate, a
non-quant friend, and a Jane Street interviewer — without three separate apps. I
asked Claude to design a `ModeProvider` context, a `body[data-mode]` attribute, a
first-visit `ModeSelector` modal, a header `ModeSwitcher`, and conditional-content
primitives `ModeText` / `ModeShow` / `ExplainBox`. Commit: `d2914b7`.

### Phase 5 — design integration (most recent, this session)

I ran Claude Design to generate a dark editorial visual system (see
`/Users/linbo/Downloads/Deep Hedging Lab 2/`). I then fed those design files back into
Claude Code with explicit instructions to preserve the audience-mode system, the
backend contract, and the browser flow. The result was a full rewrite of
`globals.css`, `layout.tsx`, the landing page, the results page, `HedgeSurface`,
`ExplainBox`, and `ModeSelector` / `ModeSwitcher`, plus four new components
(`Nav`, `Footer`, `NavMark`, `BodyModeAttr`, `PayoffShape`, `CostFrontier`,
`HeroSurfacePreview`). The integration was gated by `npm run build` + `pytest`
passing.

### Phase 6 — deployment prep (me + Claude)

I set up Vercel for the frontend and Render for the backend, added the
`NEXT_PUBLIC_API_URL` environment variable, and wrote CORS allow-origins into the
FastAPI app. Commit: `aa04d52`.

---

## What I wrote or substantially modified

- **All of `writeup/math_reference.md` and `writeup/paper_notes.md`.** These are my
  own derivations and paper annotations — I needed the math in my own hand to
  understand what I was building.
- **`spec.md` and `CLAUDE.md`.** The operating guide Claude follows every session.
  I shaped the seed-discipline rules, the "shims only, no new logic" rule, the
  locked-contract list, and the ownership-lane carve-up. These were all my decisions,
  enforced by rewriting CLAUDE.md whenever Claude drifted from them.
- **All preset config text** (`name`, `description`, `payoff_profile`,
  `benchmark_label`, `benchmark_note`) in `services/sim/main.py` — the prose that
  shows up in the UI. Claude's first drafts were either too marketing-y or too
  academic; I rewrote them to be honest about what each experiment demonstrates.
- **Several numerical-correctness fixes.** Most notably: ensuring the neural and
  classical hedgers evaluate on identical GBM paths (shared seed → `derive_seed`),
  and catching a sign error in the spread delta benchmark. These came from me
  reading the code carefully, not from Claude self-reviewing.
- **The "honest limitations" section** of the README and spec. I added this
  explicitly because I didn't want the UI to oversell what the project does — a
  portfolio that claims too much is worse than one that claims clearly.

Everything else — the bulk of the TypeScript components, the backend plumbing, the
P&L engine, the tests — was drafted by Claude from my prompts and then reviewed /
tweaked by me.

---

## Important, non-trivial prompts (quoted)

These are the prompts that materially shaped the project's direction, not the small
"rename this" or "add that field" asks.

### Prompt 1 — scoping the whole project at the outset

> "I want to build a quant-finance portfolio project targeting quant research /
> trading internships (Jane Street, Citadel, Two Sigma, Optiver, Hudson River,
> DRW). Read Buehler et al. 2019 (Deep Hedging) and propose a two-week scope I can
> realistically ship. It must demonstrate mathematical modeling, faithful paper
> implementation, disciplined engineering, and honest statistical evaluation. I do
> NOT want scope creep — call out what's out of scope loudly. Write me a `spec.md`
> and a `CLAUDE.md` as the operating guide for future Claude sessions."

### Prompt 2 — the seed-discipline rule (after a flaky test)

> "The eval paths and train paths in experiment 03 are sharing a seed and the
> neural hedger is 'recovering BS delta' only because it's seen the exact paths at
> eval. Rewrite the experiment to use derived seeds (train vs eval) and add a rule
> to CLAUDE.md: never use `np.random.seed()` globally, always pass an rng
> explicitly, and always derive child seeds from a base seed. Then rerun and show
> me the new CVaR numbers — they should be worse, and that's fine."

### Prompt 3 — migration to modular `src/` without breaking callers

> "Migrate `src/market.py`, `src/black_scholes.py`, `src/hedger.py`,
> `src/risk_measures.py`, `src/neural_hedger.py`, `src/train.py`, and
> `src/config.py` into a sub-packaged layout
> (`src/{models,payoffs,hedging,risk,neural,experiments}/`). The old top-level
> modules must become THIN SHIMS that re-export from the new locations — no new
> logic in the shims. All 34 existing tests must still pass without modification.
> If a signature has to change, call it out in a new 'locked shared contracts'
> section in CLAUDE.md."

### Prompt 4 — multi-payoff dispatch

> "The P&L engine hardcodes `call_delta` and the call payoff. I need it to accept
> an arbitrary payoff (call, put, call_spread, straddle) and the matching analytic
> delta as callable arguments. The neural hedger's feature vector must also be
> payoff-aware — currently moneyness is computed assuming a single K, but the
> spread has two strikes. Propose a feature builder interface that handles all
> four payoffs cleanly, and update the tests to cover each payoff's closed-form
> benchmark. Don't change the public `hedge_pnl` signature."

### Prompt 5 — the surface endpoint

> "I want an interactive 3D surface in the browser showing `Δ(S, τ)` for both the
> classical and neural hedgers, plus their difference. Add a
> `GET /api/runs/{id}/surface` endpoint that returns a JSON payload with
> `x` (stock price grid), `y` (time-to-maturity grid), `z_bs`, `z_neural`, `z_diff`
> (all `[n_tau][n_s]` arrays), and a metadata block including K, σ, r, T, cost, and
> the `prev_delta` slice assumption. Compute the neural surface by running the
> trained HedgeNet forward at every grid point with `prev_delta = 0`. Add a
> smoke test that the three arrays have matching shapes and that `z_diff == z_neural
> - z_bs`."

### Prompt 6 — audience-mode system (the one that took the most conversation)

> "I want the frontend to adapt to three audiences without being three separate
> apps: Beginner (non-quant friend), Guided (classmate), Expert (interviewer).
> Design this with (a) a `ModeProvider` React context that reads/writes
> `localStorage`, (b) a `body[data-mode]` attribute so CSS can key off it, (c) a
> first-visit modal that picks the mode, (d) a tiny header switcher to change it
> later, and (e) conditional-content primitives: `ModeText` (renders one of three
> React nodes), `ModeShow` (renders only in a given set of modes), and
> `ExplainBox` (a collapsible card that auto-opens in Beginner, starts collapsed
> in Guided, and is hidden in Expert unless forced). Then update the landing page
> and the run page to use these primitives. I don't want the same wall of text
> for every audience."

### Prompt 7 — design integration (this session)

> "Now use the Claude Design output from this folder as the visual source of truth
> for the web UI: `/Users/linbo/Downloads/Deep Hedging Lab 2/`. Preserve the
> current app's functionality, browser flow, audience mode system, Neural Hedge
> Surface feature, and backend contract. Keep the dark, premium, technically
> serious aesthetic. Match the design as closely as practical. Map the design
> onto: `apps/web/app/page.tsx`, `apps/web/app/runs/[id]/page.tsx`,
> `apps/web/components/*`, `apps/web/app/globals.css`. After implementing, run
> the frontend build and the backend tests, and give me a final report listing
> the design artifacts used, what you adapted, every changed page/component,
> build status, and how you confirmed the flow is preserved."

### Prompt 8 — honest results framing

> "Write an 'Honest Limitations' section for the README that calls out, explicitly,
> everything this project doesn't do: no real market data, no calibration, no
> Heston in v1, FFN not RNN, proportional cost only, single seed per UI run. I'd
> rather under-sell than over-sell — I want an interviewer to read this and trust
> the parts that ARE claimed."

### Prompt 9 — preset prose rewrite

> "Your preset descriptions sound like marketing copy. Rewrite each one in 2–3
> sentences that state exactly what the experiment tests and what the classical
> benchmark is. No hype words — 'revolutionary', 'cutting-edge', etc. If the
> neural hedger is expected to lose to classical at zero cost (it is), say so.
> Model the voice on the Buehler paper's abstract."

### Prompt 10 — final build + test gate

> "Run `npm run build` in `apps/web`, then `pytest tests/` from the repo root.
> Report pass/fail and, if anything fails, fix it without touching the public
> contracts (the `api.ts` types, the FastAPI route signatures, or the locked src/
> signatures in CLAUDE.md). If a fix requires touching those, stop and surface
> the conflict — don't silently paper over it."

---

## Commit timeline (from `git log`)

| Commit    | Date       | What it contained                                          |
|-----------|------------|------------------------------------------------------------|
| `8fa50b1` | 2026-04-23 | v0.1 flat library + 34 tests + experiments 01–05           |
| `4a7242e` | 2026-04-23 | spec.md                                                    |
| `28f1f84` | 2026-04-23 | modular src/, FastAPI backend, Next.js scaffold (big diff) |
| `131e833` | 2026-04-23 | multi-payoff dispatch, surface endpoint, new presets, tests|
| `05f01c2` | 2026-04-23 | frontend polish + simulation service extensions            |
| `b88c44a` | 2026-04-23 | phase-3 finishing touches                                  |
| `42c101e` | 2026-04-24 | stock + option hedge universe + dark-theme unification     |
| `d2914b7` | 2026-04-24 | audience-mode system (Beginner / Guided / Expert)          |
| `aa04d52` | 2026-04-24 | Vercel + Railway deployment prep                           |

Total: roughly 8 working hours spread across two calendar days, matching the
assignment's expectation. The bulk of the Python research library was drafted in
the first session; the web UI and audience modes were built out in the second.

---

## Process notes (things I would tell a future me)

1. **Write `CLAUDE.md` first and keep editing it.** Every time Claude made a
   mistake I didn't want repeated (global seeds, renaming a locked contract,
   drifting docstring style), I codified the fix into `CLAUDE.md`. This file did
   more to raise the quality ceiling than any single prompt.
2. **Use worktrees for anything risky.** The big design integration and the
   modular refactor both happened inside a git worktree. If they had gone
   sideways, `main` was untouched.
3. **Ground every AI iteration in a gate.** `pytest` and `npm run build` were the
   two things I ran after every substantive edit. When either failed, I made
   Claude fix it before moving on — never "I'll come back to it."
4. **The spec is a weapon against scope creep.** I was tempted multiple times to
   add Heston, RNN hedgers, or a calibration layer. Every time, I re-read
   `spec.md`'s "out of scope for v1" list and said no.
