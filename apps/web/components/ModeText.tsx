"use client";

import { useMode, type Mode } from "@/lib/mode";
import type { ReactNode } from "react";

interface ModeTextProps {
  beginner: ReactNode;
  guided: ReactNode;
  expert?: ReactNode;
}

/**
 * Renders different content based on the current audience mode.
 * If expert is omitted and mode is expert, renders nothing.
 */
export function ModeText({ beginner, guided, expert }: ModeTextProps) {
  const { mode } = useMode();
  if (mode === "beginner") return <>{beginner}</>;
  if (mode === "guided") return <>{guided}</>;
  return expert !== undefined ? <>{expert}</> : null;
}

interface ModeShowProps {
  /** Render content only for these modes */
  modes: Mode[];
  children: ReactNode;
}

/** Render children only if current mode is in the allowed list */
export function ModeShow({ modes, children }: ModeShowProps) {
  const { mode } = useMode();
  return modes.includes(mode) ? <>{children}</> : null;
}
