"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, type Preset } from "@/lib/api";
import { useMode } from "@/lib/mode";
import { ModeText, ModeShow } from "@/components/ModeText";
import { ExplainBox } from "@/components/ExplainBox";

export default function HomePage() {
  const router = useRouter();
  const { mode } = useMode();
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [launching, setLaunching] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getPresets()
      .then((data) => {
        setPresets(data);
        if (data.length > 0) setSelected(data[0].id);
      })
      .catch((e: Error) =>
        setFetchError(
          `Could not reach the API at ${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}. ` +
            "Make sure the backend is running.\n\n" +
            e.message
        )
      )
      .finally(() => setLoading(false));
  }, []);

  async function handleLaunch() {
    if (!selected || launching) return;
    setLaunching(true);
    try {
      const { run_id } = await api.createRun(selected);
      router.push(`/runs/${run_id}`);
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : String(e);
      alert("Failed to launch run: " + message);
      setLaunching(false);
    }
  }

  const selectedPreset = presets.find((p) => p.id === selected) ?? null;

  return (
    <div className="container" style={{ paddingBottom: 60 }}>
      <h2 className="page-title">Experiment Launcher</h2>

      {/* Subtitle — varies by mode */}
      <p className="page-subtitle">
        <ModeText
          beginner={
            <>
              Welcome. This lab shows you two ways to hedge an options position: a classical
              formula (Black-Scholes) and a neural network that learns its own strategy from
              simulated market data. Pick a scenario below and see how they compare — no prior
              quant knowledge required.
            </>
          }
          guided={
            <>
              Deep Hedging Lab compares <strong>classical delta hedging</strong> against a{" "}
              <strong>learned neural strategy</strong> that minimises CVaR of terminal P&amp;L.
              Both run on simulated GBM paths across four payoff families under proportional
              transaction costs. Choose a preset to train the neural hedger and see the
              comparison.
            </>
          }
          expert={
            <>
              Deep Hedging Lab compares <strong>classical delta hedging</strong> against a{" "}
              <strong>learned neural strategy</strong> that minimises CVaR of terminal P&amp;L.
              Both run on simulated GBM paths across four payoff families — European call, put,
              bull call spread, and straddle — under proportional transaction costs.
              Choose a preset to train the neural hedger and see the comparison.
            </>
          }
        />
      </p>

      <div className="divider" />

      {/* ── Beginner-only: concept explanations ── */}
      <ModeShow modes={["beginner"]}>
        <ExplainBox title="What is hedging?">
          <p style={{ margin: "0 0 8px" }}>
            When you sell an option — say, the right for someone to buy a stock at a fixed price
            — you take on financial risk. If the stock moves against you, you lose money.
          </p>
          <p style={{ margin: "0 0 8px" }}>
            <strong>Hedging</strong> means holding positions in other instruments (the underlying
            stock, or other options) to offset that exposure. Think of it as dynamic insurance:
            you rebalance your hedge at each time step as the market moves, so that gains in
            your hedge roughly cancel losses in the option you sold.
          </p>
          <p style={{ margin: 0 }}>
            A perfect hedge would leave you with zero net risk at all times — but in the real
            world, transaction costs, model error, and discrete rebalancing mean some residual
            risk always remains.
          </p>
        </ExplainBox>

        <ExplainBox title="What is Deep Hedging?">
          <p style={{ margin: "0 0 8px" }}>
            The classical approach uses a formula — the <strong>Black-Scholes delta</strong> —
            to decide how much of the stock to hold at each step. It works well under ideal
            assumptions, but it was not designed with transaction costs in mind.
          </p>
          <p style={{ margin: "0 0 8px" }}>
            <strong>Deep Hedging</strong> replaces the formula with a small neural network.
            The network takes market observables as input (time remaining, moneyness, current
            position) and outputs a hedge ratio. It is trained end-to-end by minimising{" "}
            <strong>CVaR</strong> — Conditional Value at Risk — which focuses the optimiser on
            protecting against the worst outcomes, not just the average.
          </p>
          <p style={{ margin: 0 }}>
            Because the network learns directly from simulated P&amp;L, it can implicitly
            account for transaction costs in a way the formula cannot.
          </p>
        </ExplainBox>

        <ExplainBox title="Why does this comparison matter?">
          <p style={{ margin: "0 0 8px" }}>
            In frictionless markets the two approaches converge — the neural hedger should
            recover something close to the Black-Scholes delta. The interesting region is when
            transaction costs are non-zero.
          </p>
          <p style={{ margin: "0 0 8px" }}>
            Classical hedging can become expensive: rebalancing aggressively to track the exact
            delta incurs costs on every trade. The neural hedger learns to trade less often or
            in smaller size when costs make frequent rebalancing counterproductive.
          </p>
          <p style={{ margin: 0 }}>
            Whether the neural strategy materially outperforms depends on the payoff, cost
            level, and market regime — which is exactly what this lab measures.
          </p>
        </ExplainBox>

        <ExplainBox title="What is a payoff? (call, put, spread, straddle)">
          <p style={{ margin: "0 0 8px" }}>
            A <strong>payoff</strong> is what you receive at the end of the contract based on
            where the stock price ends up. It defines the shape of your risk.
          </p>
          <ul style={{ margin: "0 0 8px", paddingLeft: 20 }}>
            <li style={{ marginBottom: 6 }}>
              <strong>Call</strong> — pays off if the stock finishes above the strike price.
              You profit when the market goes up.
            </li>
            <li style={{ marginBottom: 6 }}>
              <strong>Put</strong> — pays off if the stock finishes below the strike.
              You profit when the market falls.
            </li>
            <li style={{ marginBottom: 6 }}>
              <strong>Bull call spread</strong> — buy a call at one strike, sell a call at a
              higher strike. You profit from moderate upward moves, with a capped upside and
              lower cost than a plain call.
            </li>
            <li style={{ marginBottom: 0 }}>
              <strong>Straddle</strong> — buy both a call and a put at the same strike.
              You profit from large moves in either direction — useful when you expect
              volatility but are uncertain about direction.
            </li>
          </ul>
          <p style={{ margin: 0 }}>
            Each payoff has a different risk profile and a different optimal hedge shape.
            The presets below let you explore all four.
          </p>
        </ExplainBox>
      </ModeShow>

      {/* ── Guided-only: brief framing paragraph before presets ── */}
      <ModeShow modes={["guided"]}>
        <p
          style={{
            fontSize: "0.875rem",
            color: "var(--gray-500)",
            lineHeight: 1.65,
            margin: "0 0 20px",
          }}
        >
          Each preset fixes a market model, payoff, hedge universe, and transaction-cost level.
          The runner trains the neural hedger on 8 000 simulated paths per epoch, then evaluates
          both strategies on 20 000 fresh paths. Results report CVaR, mean P&amp;L, and standard
          error so you can judge statistical significance.
        </p>
      </ModeShow>

      {loading && (
        <div style={{ display: "flex", alignItems: "center", gap: 10, color: "var(--gray-400)" }}>
          <div className="spinner" />
          <span>Connecting to API…</span>
        </div>
      )}

      {fetchError && <div className="error-box">{fetchError}</div>}

      {!loading && !fetchError && (
        <>
          <p className="section-label">Choose a preset</p>

          <div className="preset-grid">
            {presets.map((preset) => (
              <div
                key={preset.id}
                className={`card${selected === preset.id ? " selected" : ""}`}
                onClick={() => setSelected(preset.id)}
                role="radio"
                aria-checked={selected === preset.id}
                tabIndex={0}
                onKeyDown={(e) => e.key === "Enter" && setSelected(preset.id)}
              >
                <div className="card-title">{preset.name}</div>
                <div className="card-desc">{preset.description}</div>
                {preset.payoff_formula && (
                  <div
                    style={{
                      fontFamily: "monospace",
                      fontSize: "0.78rem",
                      color: "var(--gray-500)",
                      marginTop: 6,
                      padding: "3px 7px",
                      background: "rgba(255,255,255,0.04)",
                      borderRadius: 4,
                      display: "inline-block",
                    }}
                  >
                    {preset.payoff_formula}
                  </div>
                )}
                <div className="card-meta">
                  <span className="badge blue">{preset.market_model}</span>
                  <span className="badge">{preset.payoff_display_name || preset.payoff_type}</span>
                  <span className={`badge ${preset.hedge_universe_display ? "blue" : ""}`}>
                    {preset.hedge_universe_display || preset.hedge_universe}
                  </span>
                  <span className={`badge ${preset.cost_rate > 0 ? "amber" : "green"}`}>
                    {preset.cost_rate > 0
                      ? `${(preset.cost_rate * 10000).toFixed(0)} bps`
                      : "Zero cost"}
                  </span>
                  <span className="badge">~{preset.est_seconds}s</span>
                </div>
              </div>
            ))}
          </div>

          {/* Selected preset detail panel */}
          {selectedPreset && (
            <div
              style={{
                margin: "20px 0",
                padding: "16px 20px",
                background: "#0f172a",
                border: "1px solid #1e293b",
                borderRadius: 8,
                fontSize: "0.84rem",
                lineHeight: 1.65,
                color: "var(--gray-400)",
              }}
            >
              <div style={{ display: "flex", flexWrap: "wrap", gap: "20px 40px" }}>
                {selectedPreset.payoff_formula && (
                  <div>
                    <span style={{ color: "var(--gray-600)", fontWeight: 600 }}>
                      Payoff:&nbsp;
                    </span>
                    <code
                      style={{
                        fontFamily: "monospace",
                        color: "#93c5fd",
                        fontSize: "0.82rem",
                      }}
                    >
                      {selectedPreset.payoff_formula}
                    </code>
                  </div>
                )}
                {selectedPreset.delta_range && (
                  <div>
                    <span style={{ color: "var(--gray-600)", fontWeight: 600 }}>
                      &Delta; range:&nbsp;
                    </span>
                    <code style={{ fontFamily: "monospace", fontSize: "0.82rem" }}>
                      {selectedPreset.delta_range}
                    </code>
                  </div>
                )}
                {selectedPreset.benchmark_label && (
                  <div>
                    <span style={{ color: "var(--gray-600)", fontWeight: 600 }}>
                      Benchmark:&nbsp;
                    </span>
                    {selectedPreset.benchmark_label}
                  </div>
                )}
              </div>
              {selectedPreset.payoff_profile && (
                <div style={{ marginTop: 8, color: "var(--gray-500)" }}>
                  {selectedPreset.payoff_profile}
                </div>
              )}
            </div>
          )}

          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <button
              className="btn btn-primary"
              onClick={handleLaunch}
              disabled={!selected || launching}
            >
              {launching ? (
                <>
                  <div className="spinner" />
                  Launching…
                </>
              ) : (
                "Launch Experiment →"
              )}
            </button>
            {selected && (
              <span style={{ fontSize: "0.85rem", color: "var(--gray-400)" }}>
                Will train for ~{presets.find((p) => p.id === selected)?.est_seconds}s
              </span>
            )}
          </div>

          <div className="divider" />

          <div style={{ fontSize: "0.85rem", color: "var(--gray-400)", lineHeight: 1.6 }}>
            <strong style={{ color: "var(--gray-600)" }}>How it works:</strong> The neural
            hedger is a feedforward network trained end-to-end via CVaR minimisation on 8 000
            simulated GBM paths per epoch. After training it is evaluated on 20 000 fresh paths.
            The classical baseline uses the payoff-specific analytic delta at each rebalancing
            step — Black-Scholes N(d₁) for calls, N(d₁)−1 for puts, net leg deltas for spreads
            and straddles — under the same cost structure.
          </div>

          {/* Beginner: annotate the "How it works" blurb for those who want more */}
          <ModeShow modes={["beginner"]}>
            <ExplainBox title="What does this mean in plain English?" defaultOpen={false}>
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                <li style={{ marginBottom: 6 }}>
                  <strong>8 000 paths per epoch</strong> — the network sees 8 000 different
                  simulated stock-price trajectories each training round. Each path is one
                  possible future for the stock.
                </li>
                <li style={{ marginBottom: 6 }}>
                  <strong>CVaR minimisation</strong> — instead of minimising average loss, the
                  network focuses on the worst 5% of outcomes. This biases it toward robustness
                  over average-case performance.
                </li>
                <li style={{ marginBottom: 6 }}>
                  <strong>20 000 fresh paths</strong> — evaluation uses paths the network never
                  saw during training, so the comparison is honest.
                </li>
                <li style={{ marginBottom: 0 }}>
                  <strong>N(d₁)</strong> — the Black-Scholes formula for how many shares of
                  stock to hold as a hedge. It is derived analytically from the option price
                  formula and changes continuously as the stock moves.
                </li>
              </ul>
            </ExplainBox>
          </ModeShow>
        </>
      )}
    </div>
  );
}
