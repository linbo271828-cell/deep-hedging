"use client";

import { useMode, type Mode } from "@/lib/mode";

const LABELS: Record<Mode, string> = {
  beginner: "Beginner",
  guided: "Guided",
  expert: "Expert",
};

export function ModeSwitcher() {
  const { mode, setMode, hasSelected } = useMode();

  // Don't flash before hydration / first selection
  if (!hasSelected) return null;

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        background: "#0f172a",
        border: "1px solid #1e293b",
        borderRadius: 20,
        padding: 2,
        gap: 0,
        marginLeft: "auto",
      }}
    >
      {(["beginner", "guided", "expert"] as Mode[]).map((m) => (
        <button
          key={m}
          onClick={() => setMode(m)}
          style={{
            padding: "4px 12px",
            borderRadius: 16,
            fontSize: "0.75rem",
            fontWeight: 500,
            border: "none",
            cursor: "pointer",
            transition: "background 0.15s, color 0.15s",
            background: mode === m ? "#1e3a5f" : "transparent",
            color: mode === m ? "#93c5fd" : "#64748b",
          }}
        >
          {LABELS[m]}
        </button>
      ))}
    </div>
  );
}
