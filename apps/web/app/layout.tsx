import type { Metadata } from "next";
import "./globals.css";

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
        <header className="header">
          <div className="header-inner">
            <h1>Deep Hedging Lab</h1>
            <span className="header-badge">v0.1 — Research Preview</span>
          </div>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
