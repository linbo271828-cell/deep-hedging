# REFLECTION.md — Deep Hedging Lab

> **Important — do not have AI write this section.**
> Per the assignment instructions: *"you must write this yourself, without AI."*
> Target length: **250–500 words total**. Be honest and specific.
> Delete this block before submitting.

---

## 1. Describe your process. At a high level, how did you go about creating this project?

_Your answer here._

<!--
Suggested things to touch on:
- Where did the idea come from? Why Buehler (2019) specifically?
- Did you start with the math, the code, the UI, or the spec?
- How did you decide what was in scope for v1 vs. what to cut?
- What order did you build things in, and why that order?
- How did you know when something was "done"?
-->

---

## 2. What AI tools and strategies did you use? What models did you choose, and were you using them in a browser chat, in something like Cursor, in the command line, etc.? Did you use an agentic process like in HW8, or something else?

_Your answer here._

<!--
Concrete things you actually used (from this project):
- Claude Code (CLI) with Opus 4.7 — agentic loop in the terminal
- Claude Design — for the visual design system
- Git worktrees for isolating risky edits
- pytest + npm run build as ground-truth gates

Mention: did you use browser chat? Cursor? Copilot? Why or why not?
Did the agentic workflow feel different from a chat window?
-->

---

## 3. Why did you make the choices above?

_Your answer here._

<!--
Why terminal-first vs. browser chat?
Why Opus 4.7 vs. a cheaper model?
Why Claude Code vs. Cursor?
Why a separate Claude Design pass for the visual system vs. designing inline?
Why CLAUDE.md as the governing document vs. just chatting every session?
-->

---

## 4. What changed from your pre-113 approach? Compare your approach to how you might have approached the project before 15-113. What has changed, if anything? Do you feel more or less well-equipped to create projects like this within a reasonable time frame?

_Your answer here._

<!--
Before 15-113 you might have:
- Written everything yourself, line by line
- Had no system for agent-governance (CLAUDE.md, locked contracts, ownership lanes)
- Not known about worktrees, pytest gates, etc.
- Either under-scoped (a toy) or over-scoped (never shipped)

Now you:
- Drive an agent with prompts and gates
- Treat the spec as the scope-control weapon
- Ship more in less wall-clock time

Are there downsides? (e.g., skills you haven't practised manually)
Where do you still feel slow or uncertain?
-->

---

## 5. What might you do with this project with more time?

_Your answer here._

<!--
Concrete extensions you considered but cut:
- Heston stochastic-volatility model (stub already exists)
- RNN / LSTM / transformer hedgers for path-dependent policies
- Calibration to real option chains (SPX, AAPL)
- Additional cost models (spread + market impact, not just proportional)
- Architecture ablations, feature ablations, objective comparisons (CVaR vs entropic)
- `reporting/` pipeline that generates the full results markdown report
- Multi-seed UI runs with error bars, not just a single-seed point estimate
- More rigorous evaluation: paired t-tests, bootstrap CIs on CVaR

Which of these would you prioritise and why?
-->
