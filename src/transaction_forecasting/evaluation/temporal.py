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
    ordered = frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="raise", utc=True)
    if ordered["timestamp"].isna().any():
        raise ValueError("timestamp cannot contain missing values")
    ordered = ordered.sort_values("timestamp", kind="stable").reset_index(drop=True)
    test_size = max(1, int(round(len(ordered) * test_fraction)))
    if test_size >= len(ordered):
        raise ValueError("At least two rows are required for a temporal split")
    # Keep all events at the boundary on the same side. Row slicing alone can
    # place identical timestamps in both sets, violating the strict time order.
    boundary = ordered.iloc[-test_size]["timestamp"]
    train = ordered[ordered["timestamp"] < boundary].copy()
    test = ordered[ordered["timestamp"] >= boundary].copy()
    if train.empty:
        raise ValueError("No nonempty strictly chronological split at this boundary")
    return train.reset_index(drop=True), test.reset_index(drop=True)
