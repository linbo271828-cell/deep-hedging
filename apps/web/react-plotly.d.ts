// Type shim so Next.js builds without errors when react-plotly.js
// types reference the global Plotly namespace from plotly.js.
declare module "react-plotly.js" {
  import * as Plotly from "plotly.js";
  import * as React from "react";

  interface PlotParams {
    data: Plotly.Data[];
    layout?: Partial<Plotly.Layout>;
    config?: Partial<Plotly.Config>;
    style?: React.CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
    onInitialized?: (figure: Plotly.Figure, graphDiv: HTMLElement) => void;
    onUpdate?: (figure: Plotly.Figure, graphDiv: HTMLElement) => void;
    onPurge?: (figure: Plotly.Figure, graphDiv: HTMLElement) => void;
    onError?: (err: Error) => void;
    [key: string]: unknown;
  }

  const Plot: React.ComponentType<PlotParams>;
  export default Plot;
}
