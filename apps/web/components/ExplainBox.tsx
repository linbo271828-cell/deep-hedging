"use client";

import { useState } from "react";
import { useMode, type Mode } from "@/lib/mode";
import type { ReactNode } from "react";

interface ExplainBoxProps {
  title: string;
  children: ReactNode;
  /** Only show in these modes (default: beginner + guided) */
  modes?: Mode[];
  /** Start expanded in beginner mode (default true) */
  defaultOpen?: boolean;
}

/**
 * A subtle collapsible explanation card. In beginner mode it opens by default;
 * in guided mode it starts collapsed. Hidden entirely in expert mode unless
 * modes prop explicitly includes "expert".
 */
export function ExplainBox({
  title,
  children,
  modes = ["beginner", "guided"],
  defaultOpen,
}: ExplainBoxProps) {
  const { mode } = useMode();
  const shouldShow = modes.includes(mode);
  const autoOpen = defaultOpen !== undefined ? defaultOpen : mode === "beginner";
  const [open, setOpen] = useState(autoOpen);

  if (!shouldShow) return null;

  return (
    <div
      style={{
        margin: "12px 0",
        border: "1px solid var(--gray-200)",
        borderRadius: "var(--radius)",
        overflow: "hidden",
        background: "rgba(15,23,42,0.6)",
      }}
    >
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          width: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "10px 14px",
          background: "transparent",
          border: "none",
          cursor: "pointer",
          color: "var(--gray-500)",
          fontSize: "0.8rem",
          fontWeight: 600,
          letterSpacing: "0.02em",
          textAlign: "left",
        }}
      >
        <span>{title}</span>
        <span style={{ fontSize: "0.7rem", opacity: 0.7 }}>{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div
          style={{
            padding: "4px 14px 14px",
            fontSize: "0.83rem",
            color: "var(--gray-500)",
            lineHeight: 1.65,
            borderTop: "1px solid var(--gray-200)",
          }}
        >
          {children}
        </div>
      )}
    </div>
  );
}
