const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Preset {
  id: string;
  name: string;
  description: string;
  cost_rate: number;
  market_model: string;
  payoff_type: string;
  hedge_universe: string;
  n_epochs: number;
  est_seconds: number;
  payoff_display_name: string;
  payoff_formula: string;
  payoff_profile: string;
  delta_range: string;
  benchmark_label: string;
  classical_description: string;
  hedge_universe_display?: string;
  hedge_universe_description?: string;
  benchmark_note?: string;
}

export interface RunStatus {
  run_id: string;
  preset_id: string;
  preset_name: string;
  status: "running" | "done" | "error";
  created_at: string;
  completed_at: string | null;
  error: string | null;
}

export interface SummaryRow {
  experiment_name: string;
  strategy: string;
  seed: number;
  cost_rate: number;
  mean_pnl: number;
  std_pnl: number;
  cvar_95: number;
  entropic_risk: number;
  expected_cost: number;
  turnover: number;
  n_paths: number;
  market_model: string;
  payoff_type: string;
  hedge_universe: string;
  n_epochs: number | null;
  hidden_dim: number | null;
  n_layers: number | null;
}

export interface RunResults {
  run_id: string;
  preset_id: string;
  preset_name: string;
  payoff_display_name: string;
  payoff_formula: string;
  payoff_profile: string;
  delta_range: string;
  benchmark_label: string;
  classical_description: string;
  hedge_universe_display?: string;
  hedge_universe_description?: string;
  benchmark_note?: string;
  neural: SummaryRow;
  classical: SummaryRow;
}

export interface SurfaceData {
  /** Stock prices along x-axis, length n_s */
  x: number[];
  /** Times to maturity along y-axis, length n_tau */
  y: number[];
  /** Black-Scholes delta surface (analytic), shape [n_tau][n_s] */
  z_bs: number[][];
  /** Neural hedge surface (prev_delta = 0 slice), shape [n_tau][n_s] */
  z_neural: number[][];
  /** Difference surface (neural − BS), shape [n_tau][n_s] */
  z_diff: number[][];
  x_label: string;
  y_label: string;
  z_label: string;
  metadata: {
    s0: number;
    k: number;
    r: number;
    sigma: number;
    T: number;
    cost_rate: number;
    payoff_type: string;
    classical_description: string;
    n_s: number;
    n_tau: number;
    z_min_bs: number;
    z_max_bs: number;
    prev_delta_assumption: string;
  };
}

export const api = {
  async getPresets(): Promise<Preset[]> {
    const res = await fetch(`${API_BASE}/api/presets`);
    if (!res.ok) throw new Error(`Failed to fetch presets: ${res.status}`);
    return res.json();
  },

  async createRun(
    presetId: string
  ): Promise<{ run_id: string; status: string }> {
    const res = await fetch(`${API_BASE}/api/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preset_id: presetId }),
    });
    if (!res.ok) throw new Error(`Failed to create run: ${res.status}`);
    return res.json();
  },

  async getRunStatus(runId: string): Promise<RunStatus> {
    const res = await fetch(`${API_BASE}/api/runs/${runId}`);
    if (!res.ok) throw new Error(`Failed to fetch run status: ${res.status}`);
    return res.json();
  },

  async getRunResults(runId: string): Promise<RunResults> {
    const res = await fetch(`${API_BASE}/api/runs/${runId}/results`);
    if (!res.ok) throw new Error(`Failed to fetch run results: ${res.status}`);
    return res.json();
  },

  getPlotUrl(runId: string): string {
    return `${API_BASE}/api/runs/${runId}/plot`;
  },

  async getSurface(runId: string): Promise<SurfaceData> {
    const res = await fetch(`${API_BASE}/api/runs/${runId}/surface`);
    if (!res.ok) throw new Error(`Failed to fetch surface data: ${res.status}`);
    return res.json();
  },
};
