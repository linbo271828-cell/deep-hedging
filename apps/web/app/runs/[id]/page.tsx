"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { api, type RunResults, type RunStatus, type SurfaceData } from "@/lib/api";
import { HedgeSurface } from "@/components/HedgeSurface";

function fmt(v: number, digits = 4): string {
  return v.toFixed(digits);
}

function MetricRow({
  label,
  neural,
  classical,
  lowerIsBetter = true,
}: {
  label: string;
  neural: number;
  classical: number;
  lowerIsBetter?: boolean;
}) {
  const neuralBetter = lowerIsBetter ? neural < classical : neural > classical;
  return (
    <tr>
      <td>{label}</td>
      <td className={`right ${neuralBetter ? "val-better" : "val-worse"}`}>
        {fmt(neural)}
      </td>
      <td className={`right ${!neuralBetter ? "val-better" : "val-worse"}`}>
        {fmt(classical)}
      </td>
    </tr>
  );
}

export default function RunPage() {
  const params = useParams();
  const router = useRouter();
  const runId = params.id as string;

  const [status, setStatus] = useState<RunStatus | null>(null);
  const [results, setResults] = useState<RunResults | null>(null);
  const [surface, setSurface] = useState<SurfaceData | null>(null);
  const [surfaceLoading, setSurfaceLoading] = useState(false);
  const [surfaceError, setSurfaceError] = useState<string | null>(null);
  const [plotLoaded, setPlotLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const s = await api.getRunStatus(runId);
      setStatus(s);
      return s.status;
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : String(e);
      setError(message);
      return "error";
    }
  }, [runId]);

  const fetchResults = useCallback(async () => {
    try {
      const r = await api.getRunResults(runId);
      setResults(r);
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : String(e);
      setError(message);
    }
  }, [runId]);

  const fetchSurface = useCallback(async () => {
    setSurfaceLoading(true);
    setSurfaceError(null);
    try {
      const s = await api.getSurface(runId);
      setSurface(s);
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : String(e);
      setSurfaceError(message);
    } finally {
      setSurfaceLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    fetchStatus().then((s) => {
      if (s === "done") {
        fetchResults();
        fetchSurface();
      }
    });
  }, [fetchStatus, fetchResults, fetchSurface]);

  useEffect(() => {
    if (!status || status.status === "done" || status.status === "error") return;

    const interval = setInterval(async () => {
      const s = await fetchStatus();
      if (s === "done") {
        clearInterval(interval);
        fetchResults();
        fetchSurface();
      } else if (s === "error") {
        clearInterval(interval);
      }
    }, 2500);

    return () => clearInterval(interval);
  }, [status?.status, fetchStatus, fetchResults, fetchSurface]);

  const runStatus = status?.status ?? "running";
  const preset = status?.preset_name ?? "—";

  return (
    <div className="container" style={{ paddingBottom: 80 }}>
      <button
        className="btn btn-ghost"
        onClick={() => router.push("/")}
        style={{ marginBottom: 24, fontSize: "0.85rem" }}
      >
        ← Back to presets
      </button>

      <h2 className="page-title">{preset}</h2>
      <div className="run-meta">Run {runId}</div>

      {/* Status indicator */}
      <div className="status-row" style={{ marginTop: 20 }}>
        <div className={`status-dot ${runStatus}`} />
        <span className="status-text">
          {runStatus === "running" && "Training & evaluating…"}
          {runStatus === "done" && "Complete"}
          {runStatus === "error" && "Error"}
        </span>
        {runStatus === "running" && (
          <div className="spinner" style={{ marginLeft: 8 }} />
        )}
        {status?.completed_at && (
          <span className="status-sub">
            Finished {new Date(status.completed_at).toLocaleTimeString()}
          </span>
        )}
        {status?.created_at && runStatus === "running" && (
          <span className="status-sub">
            Started {new Date(status.created_at).toLocaleTimeString()}
          </span>
        )}
      </div>

      {/* Error */}
      {(runStatus === "error" || error) && (
        <div className="error-box">
          {status?.error ?? error ?? "Unknown error"}
        </div>
      )}

      {/* Results */}
      {results && runStatus === "done" && (
        <>
          <div className="divider" />

          {/* Info row */}
          <div className="info-row">
            <div className="info-item">
              <span className="info-label">Market model</span>
              <span className="info-value">{results.neural.market_model.toUpperCase()}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Payoff</span>
              <span className="info-value" style={{ textTransform: "capitalize" }}>
                {results.neural.payoff_type.replace(/_/g, " ")}
              </span>
            </div>
            <div className="info-item">
              <span className="info-label">Transaction cost</span>
              <span className="info-value">
                {results.neural.cost_rate > 0
                  ? `${(results.neural.cost_rate * 10000).toFixed(0)} bps`
                  : "Zero"}
              </span>
            </div>
            <div className="info-item">
              <span className="info-label">Eval paths</span>
              <span className="info-value">{results.neural.n_paths.toLocaleString()}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Architecture</span>
              <span className="info-value">
                {results.neural.n_layers}×{results.neural.hidden_dim} FFN,{" "}
                {results.neural.n_epochs} epochs
              </span>
            </div>
          </div>

          <p className="result-section-title">Neural vs Classical — Key Metrics</p>
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th className="right">Neural Hedger</th>
                <th className="right">Classical (BS Δ)</th>
              </tr>
            </thead>
            <tbody>
              <MetricRow
                label="CVaR 95% (↓ better)"
                neural={results.neural.cvar_95}
                classical={results.classical.cvar_95}
                lowerIsBetter
              />
              <MetricRow
                label="Entropic Risk (↓ better)"
                neural={results.neural.entropic_risk}
                classical={results.classical.entropic_risk}
                lowerIsBetter
              />
              <MetricRow
                label="Mean P&L (↑ better)"
                neural={results.neural.mean_pnl}
                classical={results.classical.mean_pnl}
                lowerIsBetter={false}
              />
              <MetricRow
                label="Std P&L (↓ better)"
                neural={results.neural.std_pnl}
                classical={results.classical.std_pnl}
                lowerIsBetter
              />
              <MetricRow
                label="Expected Cost (↓ better)"
                neural={results.neural.expected_cost}
                classical={results.classical.expected_cost}
                lowerIsBetter
              />
              <MetricRow
                label="Avg Turnover (↓ better)"
                neural={results.neural.turnover}
                classical={results.classical.turnover}
                lowerIsBetter
              />
            </tbody>
          </table>

          <div className="divider" />

          <p className="result-section-title">P&L Distribution</p>
          <div className="plot-wrap">
            {!plotLoaded && (
              <div
                style={{
                  height: 200,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 10,
                  color: "var(--gray-400)",
                }}
              >
                <div className="spinner" />
                <span>Loading plot…</span>
              </div>
            )}
            <img
              src={api.getPlotUrl(runId)}
              alt="P&L distribution: neural vs classical"
              style={{ display: plotLoaded ? "block" : "none" }}
              onLoad={() => setPlotLoaded(true)}
              onError={() => setPlotLoaded(true)}
            />
            <div className="plot-caption">
              Terminal P&L distribution over {results.neural.n_paths.toLocaleString()} out-of-sample paths.
              Dashed line = mean, dotted = 5th percentile.
            </div>
          </div>

          {/* ---------------------------------------------------------------- */}
          {/* Neural Hedge Surface — signature visualization                   */}
          {/* ---------------------------------------------------------------- */}
          <div className="divider" />

          <div style={{ marginBottom: 16 }}>
            <p className="result-section-title" style={{ marginBottom: 4 }}>
              Neural Hedge Surface vs. Black-Scholes Delta
            </p>
            <p style={{ fontSize: "0.875rem", color: "var(--gray-600)", lineHeight: 1.6 }}>
              3D surfaces showing the hedge ratio Δ as a function of stock price and time to maturity.
              Switch modes to compare the learned neural policy, the analytic Black-Scholes delta,
              and their signed difference. Drag to rotate, scroll to zoom, hover for exact values.
            </p>
          </div>

          {surfaceLoading && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                color: "var(--gray-400)",
                padding: "32px 0",
              }}
            >
              <div className="spinner" />
              <span>Computing hedge surfaces…</span>
            </div>
          )}

          {surfaceError && (
            <div className="error-box" style={{ marginBottom: 16 }}>
              Surface computation failed: {surfaceError}
            </div>
          )}

          {surface && !surfaceLoading && (
            <HedgeSurface data={surface} />
          )}
        </>
      )}
    </div>
  );
}
