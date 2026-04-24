"use client";

import { createContext, useContext, useState, useEffect } from "react";

export type Mode = "beginner" | "guided" | "expert";

interface ModeCtx {
  mode: Mode;
  setMode: (m: Mode) => void;
  hasSelected: boolean;
}

export const ModeContext = createContext<ModeCtx>({
  mode: "guided",
  setMode: () => {},
  hasSelected: false,
});

export function useMode() {
  return useContext(ModeContext);
}

export function ModeProvider({ children }: { children: React.ReactNode }) {
  const [mode, setModeState] = useState<Mode>("guided");
  const [hasSelected, setHasSelected] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const saved = localStorage.getItem("dhl-mode") as Mode | null;
    const selected = localStorage.getItem("dhl-mode-selected");
    if (saved && (["beginner", "guided", "expert"] as Mode[]).includes(saved)) {
      setModeState(saved);
    }
    if (selected === "1") setHasSelected(true);
  }, []);

  function setMode(m: Mode) {
    setModeState(m);
    if (typeof window !== "undefined") {
      localStorage.setItem("dhl-mode", m);
      localStorage.setItem("dhl-mode-selected", "1");
    }
    setHasSelected(true);
  }

  // Avoid hydration mismatch: render with silent defaults until mounted
  const ctx: ModeCtx = mounted
    ? { mode, setMode, hasSelected }
    : { mode: "guided", setMode: () => {}, hasSelected: false };

  return <ModeContext.Provider value={ctx}>{children}</ModeContext.Provider>;
}
