"""Markdown report generation from saved results — STUB, not yet implemented.

Lane F (reporting/docs) owns this file.
Implement in Phase E after all experiments produce SummaryRow artifacts.

Design: the report is generated FROM saved JSON/CSV artifacts, not by re-running
experiments.  This separation means `make reproduce-report` is fast and idempotent.

Target public interface:
    generate_report(results_dir: str, output_path: str) -> None
    # Reads results/processed/summary.csv, embeds key plots, writes markdown.
"""

from __future__ import annotations


def generate_report(results_dir: str, output_path: str) -> None:
    """Generate a markdown report from saved experiment results.

    STUB — not yet implemented.
    """
    raise NotImplementedError(
        "generate_report is not yet implemented. "
        "See writeup/implementation_plan.md Phase 5."
    )
