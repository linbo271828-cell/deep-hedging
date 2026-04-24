"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, type Preset } from "@/lib/api";

export default function HomePage() {
  const router = useRouter();
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
      <p className="page-subtitle">
        Deep Hedging Lab compares <strong>classical delta hedging</strong> against a{" "}
        <strong>learned neural strategy</strong> that minimises CVaR of terminal P&amp;L.
        Both run on simulated GBM paths across four payoff families — European call, put,
        bull call spread, and straddle — under proportional transaction costs.
        Choose a preset to train the neural hedger and see the comparison.
      </p>

      <div className="divider" />

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
                  <span className="badge">{preset.hedge_universe}</span>
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
                      Δ range:&nbsp;
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
        </>
      )}
    </div>
  );
}
