"use client";

/**
 * HedgeSurface — interactive 3-mode 3D surface viewer.
 *
 * Renders three Plotly 3D surfaces from backend-computed grid data:
 *   1. Black-Scholes delta surface   (analytic, model-agnostic)
 *   2. Neural hedge surface           (learned, prev_delta = 0 slice)
 *   3. Difference surface             (neural − BS)
 *
 * Fixed-slice assumption: prev_delta is held at 0 for all grid points.
 * This shows "what the network would do starting from a flat position",
 * making the surface comparable to the classic BS delta without conditioning
 * on a particular path of prior positions.
 */

import dynamic from "next/dynamic";
import { useState } from "react";
import type { SurfaceData } from "@/lib/api";

// Dynamically import Plotly to avoid SSR / window errors in Next.js App Router.
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Mode = "bs" | "neural" | "diff";

interface ModeConfig {
  label: string;
  shortLabel: string;
  zKey: keyof Pick<SurfaceData, "z_bs" | "z_neural" | "z_diff">;
  colorscale: string;
  title: string;
  reverseScale: boolean;
  fixedZRange: [number, number] | null;
  caption: string;
}

const MODE_CONFIG: Record<Mode, ModeConfig> = {
  bs: {
    label: "Black-Scholes Δ",
    shortLabel: "BS Δ",
    zKey: "z_bs",
    colorscale: "Viridis",
    reverseScale: false,
    title: "Black-Scholes Delta Surface",
    fixedZRange: [0, 1],
    caption:
      "Analytic BS delta Δ(S, τ) = N(d₁). Exact closed-form hedge ratio at each (stock price, time to maturity) state.",
  },
  neural: {
    label: "Neural Hedge",
    shortLabel: "Neural",
    zKey: "z_neural",
    colorscale: "Viridis",
    reverseScale: false,
    title: "Neural Hedge Surface",
    fixedZRange: [0, 1],
    caption:
      "Neural hedge ratio output by the trained HedgeNet at each grid state with prev_delta = 0. " +
      "Under zero cost this should approximate BS delta; under positive cost the surface tilts " +
      "to reduce unnecessary rebalancing.",
  },
  diff: {
    label: "Difference  (Neural − BS)",
    shortLabel: "Δ Neural − BS",
    zKey: "z_diff",
    colorscale: "RdBu",
    reverseScale: true,
    title: "Difference Surface: Neural − Black-Scholes",
    fixedZRange: null,
    caption:
      "Signed deviation of the neural hedge from BS delta. " +
      "Blue regions: neural under-hedges relative to BS. " +
      "Red regions: neural over-hedges. " +
      "Under non-zero costs the network deliberately deviates near the boundary to avoid excessive friction.",
  },
};

const SCENE_STYLE = {
  bgcolor: "rgba(0,0,0,0)",
  xaxis: {
    tickfont: { color: "#64748b", size: 10 },
    gridcolor: "#1e293b",
    zerolinecolor: "#334155",
  },
  yaxis: {
    tickfont: { color: "#64748b", size: 10 },
    gridcolor: "#1e293b",
    zerolinecolor: "#334155",
  },
  zaxis: {
    tickfont: { color: "#64748b", size: 10 },
    gridcolor: "#1e293b",
    zerolinecolor: "#334155",
  },
};

export function HedgeSurface({ data }: { data: SurfaceData }) {
  const [mode, setMode] = useState<Mode>("neural");
  const cfg = MODE_CONFIG[mode];
  const z = data[cfg.zKey];

  const costLabel =
    data.metadata.cost_rate > 0
      ? `${(data.metadata.cost_rate * 10_000).toFixed(0)} bps`
      : "zero cost";

  // Compute symmetric z-range for diff mode so 0 is exactly centred
  const diffAbsMax =
    mode === "diff"
      ? Math.max(...(z as number[][]).flat().map(Math.abs))
      : null;

  const plotTrace: object = {
    type: "surface",
    x: data.x,
    y: data.y,
    z,
    colorscale: cfg.colorscale,
    reversescale: cfg.reverseScale,
    showscale: true,
    ...(cfg.fixedZRange
      ? { cmin: cfg.fixedZRange[0], cmax: cfg.fixedZRange[1] }
      : diffAbsMax != null
        ? { cmin: -diffAbsMax, cmax: diffAbsMax, zmid: 0 }
        : {}),
    colorbar: {
      thickness: 14,
      len: 0.7,
      tickfont: { color: "#94a3b8", size: 10 },
      bgcolor: "rgba(0,0,0,0)",
      bordercolor: "rgba(0,0,0,0)",
    },
    hovertemplate:
      `<b>${data.x_label}:</b> %{x:.2f}<br>` +
      `<b>${data.y_label}:</b> %{y:.3f} yr<br>` +
      `<b>Δ:</b> %{z:.5f}<extra></extra>`,
    lighting: {
      ambient: 0.7,
      diffuse: 0.7,
      specular: 0.2,
      roughness: 0.5,
    },
  };

  const layout: object = {
    title: {
      text: cfg.title,
      font: { color: "#e2e8f0", size: 14, family: "system-ui" },
      y: 0.97,
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    scene: {
      ...SCENE_STYLE,
      xaxis: {
        ...SCENE_STYLE.xaxis,
        title: { text: data.x_label, font: { color: "#94a3b8", size: 11 } },
      },
      yaxis: {
        ...SCENE_STYLE.yaxis,
        title: { text: data.y_label, font: { color: "#94a3b8", size: 11 } },
      },
      zaxis: {
        ...SCENE_STYLE.zaxis,
        title: { text: data.z_label, font: { color: "#94a3b8", size: 11 } },
        ...(cfg.fixedZRange ? { range: cfg.fixedZRange } : {}),
      },
      camera: {
        eye: { x: 1.55, y: -1.55, z: 0.75 },
        up: { x: 0, y: 0, z: 1 },
      },
      aspectratio: { x: 1.2, y: 1, z: 0.65 },
    },
    margin: { l: 0, r: 0, b: 0, t: 44 },
    font: { family: "system-ui", color: "#e2e8f0" },
  };

  return (
    <div>
      {/* Mode selector */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
        {(["bs", "neural", "diff"] as Mode[]).map((m) => (
          <button
            key={m}
            className={`btn ${mode === m ? "btn-primary" : "btn-ghost"}`}
            onClick={() => setMode(m)}
            style={{ fontSize: "0.82rem", padding: "6px 14px" }}
          >
            {MODE_CONFIG[m].shortLabel}
          </button>
        ))}
        <span
          style={{
            marginLeft: "auto",
            fontSize: "0.78rem",
            color: "var(--gray-400)",
            alignSelf: "center",
            fontStyle: "italic",
          }}
        >
          Drag to rotate · Scroll to zoom · Hover for values
        </span>
      </div>

      {/* Plotly 3D surface */}
      <div
        style={{
          background: "#0f172a",
          borderRadius: 10,
          border: "1px solid #1e293b",
          overflow: "hidden",
          boxShadow: "0 4px 24px rgba(0,0,0,0.4)",
        }}
      >
        <Plot
          data={[plotTrace] as Plotly.Data[]}
          layout={layout as Partial<Plotly.Layout>}
          config={{
            displayModeBar: true,
            modeBarButtonsToRemove: [
              "toImage",
              "sendDataToCloud",
              "select2d",
              "lasso2d",
            ] as Plotly.ModeBarButtonAny[],
            displaylogo: false,
            responsive: true,
          }}
          style={{ width: "100%", height: 480 }}
          useResizeHandler
        />
      </div>

      {/* Caption */}
      <div
        style={{
          marginTop: 12,
          padding: "10px 14px",
          background: "#0f172a",
          border: "1px solid #1e293b",
          borderRadius: 6,
          fontSize: "0.8rem",
          color: "#64748b",
          lineHeight: 1.65,
        }}
      >
        <strong style={{ color: "#94a3b8" }}>{cfg.title}.</strong>{" "}
        {cfg.caption}
        <br />
        <span style={{ marginTop: 4, display: "block" }}>
          <strong style={{ color: "#94a3b8" }}>Parameters:</strong> K ={" "}
          {data.metadata.k}, σ = {data.metadata.sigma}, r ={" "}
          {data.metadata.r}, T = {data.metadata.T} yr, cost = {costLabel}.{" "}
          <strong style={{ color: "#94a3b8" }}>Grid:</strong>{" "}
          {data.metadata.n_s} × {data.metadata.n_tau} states.{" "}
          <strong style={{ color: "#94a3b8" }}>Fixed slice:</strong>{" "}
          prev_delta = 0.
        </span>
      </div>
    </div>
  );
}
