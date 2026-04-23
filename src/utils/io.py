"""Artifact I/O utilities — STUB, partially implemented.

Owns: saving and loading experiment artifacts (configs, metrics, summaries).

Artifacts live in:
  results/raw/<experiment_name>/   — one JSON per run (config + metrics)
  results/processed/               — aggregated CSVs across runs
  results/reports/                 — generated markdown reports

Rule: plots are saved to results/reports/ by reporting/plots.py.
The raw JSON artifacts are what make experiments reproducible without re-running.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any


def save_json(data: dict[str, Any], path: str) -> None:
    """Save a dict to a JSON file, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict[str, Any]:
    """Load a JSON file and return its contents as a dict."""
    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def save_dataclass_json(obj: Any, path: str) -> None:
    """Save a dataclass instance to JSON via asdict."""
    save_json(asdict(obj), path)
