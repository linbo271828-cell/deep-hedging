import type { Metadata } from "next";
import "./globals.css";
import { ModeProvider } from "@/lib/mode";
import { ModeSelector } from "@/components/ModeSelector";
import { ModeSwitcher } from "@/components/ModeSwitcher";

export const metadata: Metadata = {
  title: "Deep Hedging Lab",
  description:
    "Compare classical Black-Scholes delta hedging against learned neural strategies across market models and transaction-cost regimes.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <ModeProvider>
          <ModeSelector />
          <header className="header">
            <div className="header-inner">
              <h1>Deep Hedging Lab</h1>
              <span className="header-badge">v0.1 — Research Preview</span>
              <ModeSwitcher />
            </div>
          </header>
          <main>{children}</main>
        </ModeProvider>
      </body>
    </html>
  );
}
