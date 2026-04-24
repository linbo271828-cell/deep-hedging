"use client";

import { useMode, type Mode } from "@/lib/mode";

const MODES: {
  id: Mode;
  label: string;
  tagline: string;
  bullets: string[];
  accent: string;
}[] = [
  {
    id: "beginner",
    label: "Beginner",
    tagline: "New to quant finance",
    bullets: [
      "Plain-English explanations throughout",
      "Every metric and chart is explained",
      "Guides you through what the results mean",
    ],
    accent: "#22c55e",
  },
  {
    id: "guided",
    label: "Guided",
    tagline: "Technical but not a quant specialist",
    bullets: [
      "Concise explanations without hand-holding",
      "Tooltips and context where it helps",
      "Balanced between signal and detail",
    ],
    accent: "#3b82f6",
  },
  {
    id: "expert",
    label: "Expert",
    tagline: "You know the domain",
    bullets: [
      "Minimal explanatory text",
      "Data-forward, compact layout",
      "No tooltips unless you ask",
    ],
    accent: "#94a3b8",
  },
];

export function ModeSelector() {
  const { hasSelected, setMode } = useMode();

  if (hasSelected) return null;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(8,13,20,0.92)",
        backdropFilter: "blur(6px)",
        zIndex: 1000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "24px",
      }}
    >
      <div
        style={{
          maxWidth: 760,
          width: "100%",
        }}
      >
        <div style={{ textAlign: "center", marginBottom: 36 }}>
          <h2
            style={{
              fontSize: "1.6rem",
              fontWeight: 700,
              color: "#f1f5f9",
              letterSpacing: "-0.03em",
              marginBottom: 10,
            }}
          >
            Choose your experience
          </h2>
          <p style={{ color: "#64748b", fontSize: "0.95rem", lineHeight: 1.55 }}>
            Deep Hedging Lab adapts to your background. You can change this anytime.
          </p>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
            gap: 16,
          }}
        >
          {MODES.map((m) => (
            <button
              key={m.id}
              onClick={() => setMode(m.id)}
              style={{
                background: "#0f172a",
                border: `1px solid #1e293b`,
                borderRadius: 10,
                padding: "24px 20px",
                cursor: "pointer",
                textAlign: "left",
                transition: "border-color 0.15s, box-shadow 0.15s",
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLButtonElement).style.borderColor = m.accent;
                (e.currentTarget as HTMLButtonElement).style.boxShadow = `0 0 0 1px ${m.accent}40`;
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLButtonElement).style.borderColor = "#1e293b";
                (e.currentTarget as HTMLButtonElement).style.boxShadow = "none";
              }}
            >
              <div
                style={{
                  display: "inline-block",
                  padding: "2px 10px",
                  borderRadius: 12,
                  fontSize: "0.78rem",
                  fontWeight: 600,
                  color: m.accent,
                  background: `${m.accent}18`,
                  border: `1px solid ${m.accent}40`,
                  marginBottom: 10,
                }}
              >
                {m.label}
              </div>
              <div
                style={{
                  fontSize: "0.82rem",
                  color: "#94a3b8",
                  marginBottom: 14,
                  lineHeight: 1.4,
                }}
              >
                {m.tagline}
              </div>
              <ul
                style={{
                  listStyle: "none",
                  padding: 0,
                  display: "flex",
                  flexDirection: "column",
                  gap: 6,
                }}
              >
                {m.bullets.map((b) => (
                  <li
                    key={b}
                    style={{
                      fontSize: "0.78rem",
                      color: "#64748b",
                      lineHeight: 1.4,
                      paddingLeft: 12,
                      position: "relative",
                    }}
                  >
                    <span
                      style={{
                        position: "absolute",
                        left: 0,
                        color: m.accent,
                        opacity: 0.8,
                      }}
                    >
                      ·
                    </span>
                    {b}
                  </li>
                ))}
              </ul>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
