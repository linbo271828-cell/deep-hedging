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

  return (
    <div className="container" style={{ paddingBottom: 60 }}>
      <h2 className="page-title">Experiment Launcher</h2>
      <p className="page-subtitle">
        Deep Hedging Lab compares <strong>classical Black-Scholes delta hedging</strong> against
        a <strong>learned neural strategy</strong> that minimises CVaR of terminal P&amp;L.
        Both run on simulated GBM paths of a short European call position.
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
                <div className="card-meta">
                  <span className="badge blue">{preset.market_model}</span>
                  <span className="badge">{preset.payoff_type}</span>
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
            <strong style={{ color: "var(--gray-600)" }}>How it works:</strong> The neural hedger
            is a feedforward network trained end-to-end via CVaR minimisation on 8 000 simulated
            GBM paths per epoch. After training it is evaluated on 20 000 fresh paths. The
            classical baseline uses exact Black-Scholes delta at each rebalancing step with the
            same cost structure.
          </div>
        </>
      )}
    </div>
  );
}
