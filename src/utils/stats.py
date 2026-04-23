"""Statistical utility functions.

Pure functions: numpy array in, scalar out.
Used for reporting standard errors alongside empirical moments.
"""

from __future__ import annotations

import numpy as np


def standard_error(arr: np.ndarray) -> float:
    """Standard error of the mean: std(arr) / sqrt(len(arr))."""
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def report_moment(arr: np.ndarray, name: str) -> str:
    """Format a one-line summary: '{name}: {mean:.4f} ± {se:.4f} (SE)'."""
    mu = float(np.mean(arr))
    se = standard_error(arr)
    return f"{name}: {mu:.4f} ± {se:.4f} (SE)"
