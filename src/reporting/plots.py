"""Plot generation from experiment results — STUB, not yet implemented.

Lane F (reporting/docs) owns this file.
Implement in Phase E (ablations + reporting).

All plots must:
  - Accept data, not file paths (no side effects from data loading).
  - Save to results/reports/ or a caller-specified path.
  - Use matplotlib only (no seaborn unless it saves significant work).
  - Minimum 150 dpi, (8,5) single-panel or (12,5) side-by-side.
  - Labelled axes, title, legend when > 1 series.

Target functions:
    plot_cost_frontier(rows, output_path) -> None
    plot_pnl_distribution(pnl_dict, output_path) -> None
    plot_learned_vs_bs_delta(learned, bs, output_path) -> None
"""

from __future__ import annotations


def plot_cost_frontier(rows: list[object], output_path: str) -> None:
    """Plot CVaR vs. expected transaction cost across cost-rate sweep.

    STUB — not yet implemented.
    """
    raise NotImplementedError(
        "plot_cost_frontier is not yet implemented. "
        "See writeup/implementation_plan.md Phase 5."
    )
