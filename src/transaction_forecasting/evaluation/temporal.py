"""Leakage-safe temporal evaluation helpers."""

from __future__ import annotations

import pandas as pd


def temporal_split(
    frame: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows chronologically; the test set is strictly after train."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if "timestamp" not in frame:
        raise ValueError("temporal_split requires a timestamp column")
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    test_size = max(1, int(round(len(ordered) * test_fraction)))
    if test_size >= len(ordered):
        raise ValueError("At least two rows are required for a temporal split")
    return ordered.iloc[:-test_size].copy(), ordered.iloc[-test_size:].copy()
